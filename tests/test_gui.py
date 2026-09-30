import csv
import types
import dataclasses
import io
import json
import os
import shutil
import time

import matplotlib.colors
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractSpinBox
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFileDialog, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSplitter, QTableWidgetItem,
)

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import (
    MainWindow, _DosySettingsDialog, _GradientCalibrationDialog,
    _ProcessingReportDialog, _ScoreSettingsDialog,
)
from opennmr.gui.session import Session
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.gui.widgets.collapsible_section import CollapsibleSection
import opennmr.gui.main_window as main_window_module
from opennmr import io as opennmr_io
from opennmr.io import bruker as io_bruker
from opennmr.io import persistence as io_persistence
from opennmr.processing import diffusion as processing_diffusion
from opennmr.processing import fitting as processing_fitting
from opennmr.processing import ilt as processing_ilt
from opennmr.processing import score as processing_score
from opennmr.processing import relaxation as processing_relaxation
from opennmr.processing import slicing as processing_slicing
from opennmr.processing import ft as processing_ft
from opennmr.processing.slicing import get_trace
from opennmr.processing.fitting import gaussian, lorentzian


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _set_delay_unit(window, unit: str) -> None:
    """Put the Curve Fitting panel's Unit combo on ``unit`` ("s", "ms"
    or "µs"). Tests that care about absolute delays set it explicitly
    rather than relying on the default, so what a typed number means is
    visible in the test itself."""
    index = window.delay_unit_combo.findText(unit)
    assert index >= 0, f"no {unit!r} unit in the combo"
    window.delay_unit_combo.setCurrentIndex(index)


def _write_params(path, params):
    lines = []
    for key, value in params.items():
        if isinstance(value, str):
            lines.append(f"##${key}= <{value}>")
        else:
            lines.append(f"##${key}= {value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _make_1d_dataset(tmp_path, n=8, name="expt1"):
    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.13, "SFO1": 400.13, "O1": 2000.0, "NS": 4})
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n, "SF": 400.13, "NC_proc": 0})
    (np.arange(n, dtype="int32") - n // 2).tofile(pdata / "1r")
    np.zeros(n, dtype="int32").tofile(pdata / "1i")
    return str(expt)


def _make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="expt2d", f1_nuc="1H", with_imag=True):
    # `acqu2s`'s NUC1 deliberately real (not "off"), matching a pulse
    # sequence (e.g. diffusion, relaxation) whose F1 isn't genuinely a
    # frequency dimension even though the table says otherwise --
    # exactly the case the `pseudo_2d` override exists for.
    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1})
    _write_params(expt / "acqu2s", {"NUC1": f1_nuc, "BF1": 400.13, "O1": 0.0})
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 400.13})
    data = np.arange(n_f1 * n_f2, dtype="int32").reshape(n_f1, n_f2)
    data.tofile(pdata / "2rr")
    if with_imag:
        (-data).tofile(pdata / "2ri")
    return str(expt)


def test_main_window_constructs(qapp):
    window = MainWindow()
    assert window.session is not None
    assert window.spectrum_list.count() == 0


def test_main_window_side_tabs_scroll_instead_of_growing_the_window(qapp):
    # Regression test: the side tabs (Processing/Analysis especially)
    # show/hide whole groups of controls per selected spectrum. Qt
    # grows a top-level window to fit a bigger layout minimum size but
    # never auto-shrinks it back down -- reported as the window
    # ratcheting taller over a session. Wrapping each tab's content in
    # a QScrollArea (see MainWindow._scrollable) keeps that size-hint
    # growth from propagating up to the window at all.
    from PySide6.QtWidgets import QScrollArea

    window = MainWindow()
    left_tabs = window.side_tabs
    for i in range(left_tabs.count()):
        page = left_tabs.widget(i)
        assert isinstance(page, QScrollArea)
        assert page.widgetResizable() is True


def test_main_window_no_long_unwrapped_labels_in_left_tabs(qapp):
    # Systemic guard: a QLabel with long text and no setWordWrap(True)
    # reports its full single-line text as its sizeHint width, which
    # propagates up through the layout and forces the whole side panel
    # wider -- happened twice now (the Compare tab's own intro label,
    # fixed earlier; then two labels added to the VOCS sub-tab this
    # session, missed the same way). A short label legitimately
    # doesn't need wrapping (e.g. "Zero-fill to:"), so this only flags
    # long ones -- long enough that an unwrapped single line would
    # visibly force the panel wide.
    window = MainWindow()
    offenders = []
    for label in window.findChildren(QLabel):
        text = label.text()
        if len(text) > 60 and not label.wordWrap():
            offenders.append(text)
    assert offenders == []


def test_scan_for_spectra_finds_single_bruker_experiment_picked_directly(tmp_path):
    path = _make_1d_dataset(tmp_path)
    matches = MainWindow._scan_for_spectra(path)
    assert matches == [(os.path.abspath(path), "bruker")]


def test_scan_for_spectra_does_not_descend_into_a_recognised_experiment(tmp_path):
    # A Bruker experiment's own pdata/1/procs etc. must not also be
    # reported as separate matches once the experiment root itself is
    # recognised.
    path = _make_1d_dataset(tmp_path)
    matches = MainWindow._scan_for_spectra(path)
    assert len(matches) == 1


def test_scan_for_spectra_finds_multiple_numbered_experiments(tmp_path):
    a = _make_1d_dataset(tmp_path, name="1")
    b = _make_1d_dataset(tmp_path, name="2")
    matches = MainWindow._scan_for_spectra(str(tmp_path))
    found_paths = {os.path.abspath(p) for p, _v in matches}
    assert found_paths == {os.path.abspath(a), os.path.abspath(b)}
    assert all(v == "bruker" for _p, v in matches)


def test_scan_for_spectra_orders_numbered_experiments_naturally(tmp_path):
    for name in ("2", "1", "11", "10", "3"):
        _make_1d_dataset(tmp_path, name=name)
    matches = MainWindow._scan_for_spectra(str(tmp_path))
    names = [os.path.basename(p) for p, _v in matches]
    assert names == ["1", "2", "3", "10", "11"]


def test_scan_for_spectra_finds_xpulse_file_directly(tmp_path):
    jdx = tmp_path / "spectrum.jdx"
    jdx.write_text("##TITLE= test\n##$SFO1= 400.13\n")
    matches = MainWindow._scan_for_spectra(str(tmp_path))
    assert matches == [(str(jdx), "xpulse")]


def test_scan_for_spectra_finds_nested_xpulse_slice_files(tmp_path):
    # Mirrors a real X-Pulse pseudo-2D layout: raw_slices/NNNN_slice.jdx
    # -- individual slices several levels deep must still be found.
    # (Named "_slice.jdx", not "_raw.jdx" -- see
    # test_scan_for_spectra_skips_individual_raw_scan_files for the
    # sibling case this must NOT be confused with.)
    slice_dir = tmp_path / "expt" / "raw_slices"
    slice_dir.mkdir(parents=True)
    jdx = slice_dir / "0001_slice.jdx"
    jdx.write_text("##TITLE= test\n##$SFO1= 400.13\n")
    matches = MainWindow._scan_for_spectra(str(tmp_path))
    assert matches == [(str(jdx), "xpulse")]


def test_scan_for_spectra_skips_individual_raw_scan_files(tmp_path):
    # Real X-Pulse layout: an averaged spectrum alongside per-scan
    # "..._raw.jdx" files under raw/ -- only the averaged one should
    # be offered, not N near-duplicate individual scans.
    expt = tmp_path / "expt"
    raw_dir = expt / "raw"
    raw_dir.mkdir(parents=True)
    avg = expt / "sample_avg.jdx"
    avg.write_text("##TITLE= test\n##$SFO1= 400.13\n")
    for i in range(1, 5):
        (raw_dir / f"sample_{i}_raw.jdx").write_text("##TITLE= test\n##$SFO1= 400.13\n")

    matches = MainWindow._scan_for_spectra(str(tmp_path))
    assert matches == [(str(avg), "xpulse")]


def test_scan_for_spectra_unrecognised_folder_returns_empty(tmp_path):
    (tmp_path / "not_nmr_data.txt").write_text("hello")
    assert MainWindow._scan_for_spectra(str(tmp_path)) == []


def test_scan_for_spectra_skips_raw_slices_redundant_with_an_ntuples_2d_file(tmp_path):
    # Real reported symptom: a genuine X-Pulse 2D dataset (e.g. HSQC)
    # has both its own real NTUPLES "..._avg.jdx" file *and* a sibling
    # raw_slices/ folder holding the same data per-F1-increment --
    # scanning used to offer 129 disconnected matches (128 individual
    # slices + the one real dataset) instead of recognizing it as one.
    expt = tmp_path / "expt"
    slice_dir = expt / "raw_slices"
    slice_dir.mkdir(parents=True)
    for i in range(1, 5):
        (slice_dir / f"{i:04d}_slice.jdx").write_text("##TITLE= slice\n##$SFO1= 400.13\n")
    ntuples = expt / "sample_avg.jdx"
    ntuples.write_text("##TITLE= test\n##DATACLASS= NTUPLES\n##NUMDIM= 2\n##$SFO1= 400.13\n")

    matches = MainWindow._scan_for_spectra(str(tmp_path))

    assert matches == [(str(ntuples), "xpulse")]


def test_scan_for_spectra_keeps_raw_slices_without_an_ntuples_sibling(tmp_path):
    # The same folder *name* ("raw_slices") is also used for genuine,
    # standalone per-slice pseudo-2D data (e.g. a PFG series) with no
    # NTUPLES file anywhere nearby -- must stay fully included, not
    # suppressed just because the folder happens to be named the same.
    slice_dir = tmp_path / "expt" / "raw_slices"
    slice_dir.mkdir(parents=True)
    for i in range(1, 4):
        (slice_dir / f"{i:04d}_slice.jdx").write_text("##TITLE= slice\n##$SFO1= 400.13\n")

    matches = MainWindow._scan_for_spectra(str(tmp_path))

    assert len(matches) == 3


def test_main_window_open_clicked_loads_single_match_directly(qapp, tmp_path, monkeypatch):
    path = _make_1d_dataset(tmp_path)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: path)

    window = MainWindow()
    window._on_open_clicked()

    assert window.spectrum_list.count() == 1
    assert window.session.spectra[0].metadata.vendor == "bruker"


def test_main_window_open_clicked_prompts_when_multiple_matches(qapp, tmp_path, monkeypatch):
    _make_1d_dataset(tmp_path, name="1")
    _make_1d_dataset(tmp_path, name="2")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))

    window = MainWindow()
    prompted = {}

    def fake_prompt(matches, base_dir):
        prompted["matches"] = matches
        return [matches[0]]  # simulate picking the first one

    monkeypatch.setattr(window, "_prompt_select_spectra", fake_prompt)
    window._on_open_clicked()

    assert len(prompted["matches"]) == 2
    assert window.spectrum_list.count() == 1


def test_main_window_open_clicked_no_matches_shows_error(qapp, tmp_path, monkeypatch):
    (tmp_path / "junk.txt").write_text("hello")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))

    shown = {}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))

    window = MainWindow()
    window._on_open_clicked()

    assert shown.get("called") is True
    assert window.spectrum_list.count() == 0


def test_main_window_open_files_clicked_loads_each_file(qapp, tmp_path, monkeypatch):
    path = _make_xpulse_dataset(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([path], ""))

    window = MainWindow()
    window._on_open_files_clicked()

    assert window.spectrum_list.count() == 1
    assert window.session.spectra[0].metadata.vendor == "xpulse"


def test_main_window_open_files_clicked_cancelled_does_nothing(qapp, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))

    window = MainWindow()
    window._on_open_files_clicked()

    assert window.spectrum_list.count() == 0


def test_main_window_open_files_reports_unrecognised_files(qapp, tmp_path, monkeypatch):
    good = _make_xpulse_dataset(tmp_path)
    junk = tmp_path / "junk.txt"
    junk.write_text("hello")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([good, str(junk)], ""))

    shown = {}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))

    window = MainWindow()
    window._on_open_files_clicked()

    assert window.spectrum_list.count() == 1
    assert shown.get("called") is True


def test_main_window_accepts_drops(qapp):
    window = MainWindow()
    assert window.acceptDrops() is True


def test_main_window_drop_directory_loads_single_experiment(qapp, tmp_path):
    path = _make_1d_dataset(tmp_path)
    window = MainWindow()
    window._handle_dropped_paths([path])
    assert window.spectrum_list.count() == 1
    assert window.session.spectra[0].metadata.vendor == "bruker"


def test_main_window_drop_directory_with_multiple_experiments_prompts(qapp, tmp_path, monkeypatch):
    _make_1d_dataset(tmp_path, name="1")
    _make_1d_dataset(tmp_path, name="2")

    window = MainWindow()
    prompted = {}

    def fake_prompt(matches, base_dir):
        prompted["matches"] = matches
        return [matches[0]]

    monkeypatch.setattr(window, "_prompt_select_spectra", fake_prompt)
    window._handle_dropped_paths([str(tmp_path)])

    assert len(prompted["matches"]) == 2
    assert window.spectrum_list.count() == 1


def test_main_window_drop_single_xpulse_file_loads_directly(qapp, tmp_path):
    path = _make_xpulse_dataset(tmp_path)
    window = MainWindow()
    window._handle_dropped_paths([path])
    assert window.spectrum_list.count() == 1
    assert window.session.spectra[0].metadata.vendor == "xpulse"


def test_main_window_drop_unrecognised_file_is_skipped(qapp, tmp_path):
    junk = tmp_path / "junk.txt"
    junk.write_text("hello")
    window = MainWindow()
    window._handle_dropped_paths([str(junk)])
    assert window.spectrum_list.count() == 0


def test_main_window_drop_multiple_paths_loads_each(qapp, tmp_path):
    path_a = _make_1d_dataset(tmp_path, name="a")
    path_b = _make_1d_dataset(tmp_path, name="b")
    window = MainWindow()
    window._handle_dropped_paths([path_a, path_b])
    assert window.spectrum_list.count() == 2


def test_session_add_and_label(tmp_path):
    session = Session()
    path = _make_1d_dataset(tmp_path)
    session.add(path, vendor="bruker", mass=1.0)
    assert len(session) == 1
    label = session.label(0)
    assert "1H" in label
    assert "bruker" in label


def test_session_remove_and_clear(tmp_path):
    session = Session()
    session.add(_make_1d_dataset(tmp_path, n=8, name="expt_a"), vendor="bruker")
    session.add(_make_1d_dataset(tmp_path, n=8, name="expt_b"), vendor="bruker")
    assert len(session) == 2
    session.remove(0)
    assert len(session) == 1
    session.clear()
    assert len(session) == 0


def test_session_set_nucleus_override(tmp_path):
    session = Session()
    session.add(_make_1d_dataset(tmp_path), vendor="bruker")
    session.set_nucleus(0, "7Li")
    assert session.spectra[0].metadata.nucleus == "7Li"
    assert "7Li" in session.label(0)


def test_session_suggest_nucleus_500mhz_console_on_100mhz_magnet(tmp_path):
    session = Session()
    session.add(_make_1d_dataset(tmp_path), vendor="bruker")
    # Simulate a nuclide picked from a table calibrated for an 11.75T
    # (500 MHz) console, actually run on a 2.35T (100 MHz) magnet: the
    # recorded SFO1 corresponds to 7Li's real gyromagnetic ratio once
    # the field mismatch is accounted for.
    session.set_nucleus(0, "189Os")
    import dataclasses
    session.spectra[0] = dataclasses.replace(
        session.spectra[0],
        metadata=dataclasses.replace(session.spectra[0].metadata, sfo1=16.54845 * 2.35),
    )
    assert session.suggest_nucleus(0) == "7Li"


def test_session_suggest_nucleus_returns_none_without_sfo1(tmp_path):
    session = Session()
    session.add(_make_1d_dataset(tmp_path), vendor="bruker")
    import dataclasses
    session.spectra[0] = dataclasses.replace(
        session.spectra[0],
        metadata=dataclasses.replace(session.spectra[0].metadata, sfo1=None),
    )
    assert session.suggest_nucleus(0) is None


def test_session_reload_with_pseudo_2d_override(tmp_path):
    session = Session()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8, f1_nuc="1H")
    session.add(path, vendor="bruker")

    assert session.spectra[0].is_pseudo_2d is False
    assert session.spectra[0].axes[0].kind == "frequency"

    session.reload(0, pseudo_2d=True)
    assert session.spectra[0].is_pseudo_2d is True
    assert session.spectra[0].axes[0].kind == "index"
    # Data content is unchanged -- only the F1 axis interpretation.
    np.testing.assert_array_equal(session.spectra[0].data, np.arange(32).reshape(4, 8))


def test_session_reload_preserves_nucleus_override(tmp_path):
    session = Session()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    session.add(path, vendor="bruker")
    session.set_nucleus(0, "7Li")

    session.reload(0, pseudo_2d=True)
    assert session.spectra[0].metadata.nucleus == "7Li"


def test_session_reload_remembers_override_on_later_reload(tmp_path):
    session = Session()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    session.add(path, vendor="bruker")
    session.reload(0, pseudo_2d=True)
    # Reloading again without re-specifying pseudo_2d should keep it
    # (e.g. after a nucleus override triggers no reload, the pseudo_2d
    # choice made earlier must still stick).
    session.reload(0)
    assert session.spectra[0].is_pseudo_2d is True


def test_main_window_load_and_plot(qapp, tmp_path):
    window = MainWindow()
    path = _make_1d_dataset(tmp_path)
    window._load(path, vendor="bruker")
    assert window.spectrum_list.count() == 1
    assert window.spectrum_list.currentRow() == 0
    # Selection triggers a plot; canvas should now have a trace drawn.
    assert len(_trace_lines(window.canvas.ax)) == 1


def test_main_window_process_indirect_button_hidden_for_1d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.process_indirect_button.isHidden() is True


# ============================================================
# Menu bar
# ============================================================


def _menu_by_title(window, title):
    """A top-level menu by its own name. Positional lookup broke every
    time a menu was added -- name lookup does not."""
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    raise AssertionError(f"no {title!r} menu")


def test_main_window_has_file_edit_window_menus(qapp):
    window = MainWindow()
    titles = [action.text().replace("&", "") for action in window.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Window", "Analysis", "Fitting"]


def test_main_window_menu_open_action_triggers_open_dialog(qapp, monkeypatch):
    window = MainWindow()
    called = []
    monkeypatch.setattr(window, "_on_open_clicked", lambda: called.append(True))
    file_menu = window.menuBar().actions()[0].menu()
    open_action = next(a for a in file_menu.actions() if "Open" in a.text())
    open_action.trigger()
    assert called == [True]


def test_main_window_menu_open_files_action_triggers_open_files_dialog(qapp, monkeypatch):
    window = MainWindow()
    called = []
    monkeypatch.setattr(window, "_on_open_files_clicked", lambda: called.append(True))
    file_menu = window.menuBar().actions()[0].menu()
    open_files_action = next(a for a in file_menu.actions() if "File(s)" in a.text())
    open_files_action.trigger()
    assert called == [True]


def test_main_window_data_tab_has_no_open_button(qapp):
    """The Data tab's Open… button was removed once the folder browser
    made it redundant ("the open button should be relegated to a menu
    item and removed from the data tab now we have the folder
    browser").

    Both dialogs it fronted have always also been File-menu actions
    with their own shortcuts, so this asserts the menu still offers
    them -- removing the button is only safe while that holds."""
    window = MainWindow()

    assert not hasattr(window, "open_button")

    file_menu = window.menuBar().actions()[0].menu()
    labels = [action.text() for action in file_menu.actions()]
    assert any("Open" in label and "File(s)" not in label for label in labels)
    assert any("File(s)" in label for label in labels)


def test_main_window_menu_remove_action_removes_selected_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.spectrum_list.count() == 1

    edit_menu = window.menuBar().actions()[1].menu()
    remove_action = next(a for a in edit_menu.actions() if "Remove" in a.text())
    remove_action.trigger()

    assert window.spectrum_list.count() == 0


def test_main_window_remove_spectrum_shortcut_is_scoped_to_spectrum_list(qapp):
    # Delete/Backspace used to remove the selected spectrum no matter
    # which widget had focus (QAction's default WindowShortcut context),
    # which made it too easy to trigger by accident from, say, the
    # Analysis tab. It must now only fire while the Data tab's own list
    # is focused -- the menu item itself stays clickable from anywhere.
    window = MainWindow()
    edit_menu = window.menuBar().actions()[1].menu()
    remove_action = next(a for a in edit_menu.actions() if "Remove Selected Spectrum" in a.text())
    assert remove_action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
    assert remove_action in window.spectrum_list.actions()


def test_main_window_remove_peak_shortcut_is_scoped_to_peak_table(qapp, tmp_path):
    # Backspace should remove a selected row from the peak table when
    # the Analysis tab's peak table has focus, instead of the global
    # remove-spectrum shortcut firing.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._add_peak_row(shift=1.0, fwhm=0.5, eta=0.5)
    remove_peak_action = next(a for a in window.peak_table.actions() if "Peak" in a.text())
    assert remove_peak_action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut

    window.peak_table.setCurrentCell(0, 0)
    remove_peak_action.trigger()

    assert window.peak_table.rowCount() == 0
    assert window.spectrum_list.count() == 1


def _side_tab_names(window):
    """The launcher tiles under their descriptive names (the Window
    menu's), in tile order: the tiles themselves carry short labels
    ("Open", "Dynamics", ...), which the structural tests below should
    not have to track. Tiles without a menu entry keep their label."""
    names = {tile: label for label, tile in MainWindow._WINDOW_MENU_PANELS}
    return [names.get(window.side_tabs.tabText(i), window.side_tabs.tabText(i))
            for i in range(window.side_tabs.count())]


def test_main_window_menu_window_panel_actions_switch_side_tabs(qapp):
    window = MainWindow()
    window_menu = _menu_by_title(window, "Window")
    applications_action = next(a for a in window_menu.actions() if a.text() == "Applications")
    applications_action.trigger()
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Apply"


def test_every_window_menu_panel_entry_reaches_its_tile(qapp):
    # The tiles were renamed to short labels ("Open", "Dynamics", ...)
    # while the menu still looked them up by the old names, so every
    # Panel entry silently did nothing. Each must land on a real tile,
    # and the tiles must appear in the menu in their own order.
    window = MainWindow()
    tiles = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    window_menu = _menu_by_title(window, "Window")
    for label, tile in MainWindow._WINDOW_MENU_PANELS:
        assert tile in tiles, tile
        window.side_tabs.setCurrentIndex(tiles.index("Settings"))
        next(a for a in window_menu.actions() if a.text() == label).trigger()
        assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == tile
    assert [t for _l, t in MainWindow._WINDOW_MENU_PANELS] == tiles[:len(MainWindow._WINDOW_MENU_PANELS)]


def test_window_menu_shortcuts_are_unique_and_valid(qapp):
    # Spectrum View repeated Simulations' Ctrl+7 (Qt fires neither) and
    # three views got "Ctrl+10".."Ctrl+12", which are not shortcuts.
    # Upstream (Leo) has the equivalent uniqueness test for his menu.
    window = MainWindow()
    window_menu = _menu_by_title(window, "Window")
    actions = [a for a in window_menu.actions() if not a.isSeparator() and a.text()]
    shortcuts = [a.shortcut().toString() for a in actions if not a.shortcut().isEmpty()]
    assert "" not in shortcuts
    assert len(shortcuts) == len(set(shortcuts)), shortcuts
    views = [a for a in actions if a.text().endswith(" View")]
    assert [a.shortcut().toString() for a in views] == [f"Ctrl+Shift+{n}" for n in range(1, 7)]


def test_main_window_spectrum_canvas_defaults_to_full_resolution_others_do_not(qapp):
    # Explicit user feedback: the main Spectrum tab was silently,
    # severely downsampled by default -- "only the spectrum in fid
    # processing should be [downsampled]... I need to have the fully
    # sampled spectrum in the spectrum tab." Every other canvas keeps
    # the fast/approximate default, which is the right tradeoff there
    # (explicitly named: FID processing).
    #
    # Compare joined the Spectrum tab on 2026-09-28 (explicit user
    # request: "we should always see the full, true data (as in
    # 'Spectrum') when in the Compare window viewer").
    window = MainWindow()
    assert window.canvas._full_resolution is True
    assert window.compare_canvas._full_resolution is True
    assert window.processing_preview_canvas._full_resolution is False
    assert window.fid_canvas._full_resolution is False
    assert window.fit_canvas._full_resolution is False
    assert window.spectral_fit_canvas._full_resolution is False


def test_main_window_menu_window_view_actions_switch_canvas_tabs(qapp):
    window = MainWindow()
    window_menu = _menu_by_title(window, "Window")
    compare_action = next(a for a in window_menu.actions() if a.text() == "Compare View")
    compare_action.trigger()
    assert window.canvas_tabs.currentWidget() is window.compare_canvas


def test_main_window_menu_save_figure_delegates_to_current_toolbar(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    called = []
    monkeypatch.setattr(window.canvas.toolbar, "save_figure", lambda: called.append(True))
    window._on_save_figure_clicked()
    assert called == [True]


def test_main_window_process_indirect_button_reflects_2d_state(qapp, tmp_path):
    # _make_2d_dataset's own vendor-recorded metadata would normally
    # auto-detect as conventional 2D, but pseudo-2D is now the default
    # for any 2D data regardless (see _load) -- so the button starts
    # visible-but-unchecked (not processing the indirect dimension)
    # and the spectrum itself is already pseudo-2D straight off _load,
    # with no click needed.
    window = MainWindow()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    window._load(path, vendor="bruker")
    index = window._selected_index()
    assert window.process_indirect_button.isHidden() is False
    assert window.process_indirect_button.isChecked() is False
    assert window.session.spectra[index].is_pseudo_2d is True


def test_main_window_process_indirect_button_reprocesses_and_replots(qapp, tmp_path):
    window = MainWindow()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    window._load(path, vendor="bruker")
    index = window._selected_index()

    window.process_indirect_button.setChecked(True)

    assert window.session.spectra[index].is_pseudo_2d is False
    assert window.session.spectra[index].axes[0].kind == "frequency"
    # Canvas re-plotted with the new interpretation.
    assert window.canvas.ax is not None


def test_main_window_process_indirect_button_toggles_back_to_pseudo2d(qapp, tmp_path):
    # Regression test: every other process_indirect_button test only
    # ever exercises pseudo-2D -> conventional (unchecked -> checked).
    # The reverse direction (conventional -> pseudo-2D, unchecking a
    # previously-checked button) must work too -- explicit user ask
    # ("a way to switch... in both directions").
    window = MainWindow()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    window._load(path, vendor="bruker")
    index = window._selected_index()

    window.process_indirect_button.setChecked(True)
    assert window.session.spectra[index].is_pseudo_2d is False
    assert window.session.spectra[index].axes[0].kind == "frequency"

    window.process_indirect_button.setChecked(False)
    assert window.session.spectra[index].is_pseudo_2d is True
    assert window.session.spectra[index].axes[0].kind == "index"
    assert window.canvas.ax is not None

    # And back again, to confirm it isn't a one-shot toggle.
    window.process_indirect_button.setChecked(True)
    assert window.session.spectra[index].is_pseudo_2d is False


def test_main_window_process_indirect_button_preserves_nucleus_override(qapp, tmp_path):
    window = MainWindow()
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    window._load(path, vendor="bruker")
    index = window._selected_index()

    window._set_nucleus(index, "7Li")

    window.process_indirect_button.setChecked(True)
    assert window.session.spectra[index].metadata.nucleus == "7Li"


def test_main_window_remove_button(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    assert window.spectrum_list.count() == 2

    window._select_single_row(0)
    window._on_remove_clicked()
    assert window.spectrum_list.count() == 1
    assert len(window.session) == 1
    # Several selected: Remove removes them all (2026-09-30).
    window._load(_make_1d_dataset(tmp_path, name="c"), vendor="bruker")
    window.spectrum_list.selectAll()
    window._on_remove_clicked()
    assert window.spectrum_list.count() == 0 and len(window.session) == 0


def test_main_window_clear_button(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")

    window._on_clear_clicked()
    assert window.spectrum_list.count() == 0
    assert len(window.session) == 0
    assert window.phase0_handle.isEnabled() is False


def test_main_window_nucleus_override(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    window._set_nucleus(0, "7Li")

    assert window.session.spectra[0].metadata.nucleus == "7Li"
    assert "7Li" in window.spectrum_list.item(0).text()
    assert "7Li" in window.info_label.text()


def test_main_window_override_nucleus_action_uses_dialog(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("7Li", True))
    window._on_override_nucleus_action(0)

    assert window.session.spectra[0].metadata.nucleus == "7Li"


def test_main_window_override_nucleus_action_cancelled_is_noop(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("7Li", False))
    window._on_override_nucleus_action(0)

    assert window.session.spectra[0].metadata.nucleus == "1H"


def test_main_window_suggest_nucleus_action_applies_on_confirm(qapp, tmp_path, monkeypatch):
    import dataclasses

    from PySide6.QtWidgets import QMessageBox

    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.session.set_nucleus(0, "189Os")
    window.session.spectra[0] = dataclasses.replace(
        window.session.spectra[0],
        metadata=dataclasses.replace(window.session.spectra[0].metadata, sfo1=16.54845 * 2.35),
    )
    window._refresh_selection_display(0)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window._on_suggest_nucleus_action(0)

    assert window.session.spectra[0].metadata.nucleus == "7Li"


def test_main_window_spectrum_list_context_menu_wires_actions(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    # Just confirm the handler resolves an item and doesn't crash when
    # there's nothing under the cursor -- QMenu.exec() itself is modal
    # and not meaningfully testable headlessly.
    from PySide6.QtCore import QPoint

    window._on_spectrum_list_context_menu(QPoint(-1, -1))


def test_main_window_processing_enabled_for_1d_with_imag(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.phase0_handle.isEnabled() is True
    assert window.pivot_spin.isEnabled() is True


def test_main_window_phase_handle_updates_canvas_live(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    index = window._selected_index()
    # This is about phase reset, not baseline -- disable the
    # now-on-by-default baseline correction so it doesn't flatten this
    # fixture's own linear-ramp data (indistinguishable from a pure
    # baseline) before phase ever gets a chance to change anything.
    window.baseline_correct_button.setChecked(False)
    original_real = window.session.spectra[index].data.copy()

    window.phase0_handle.setValue(90.0)
    assert window.phase0_spin.value() == pytest.approx(90.0)

    adjusted_real = window.session.spectra[index].data
    assert not np.allclose(original_real, adjusted_real)

    # Reset returns to (approximately) the original state.
    window._on_reset_phase_clicked()
    np.testing.assert_allclose(window.session.spectra[index].data, original_real, atol=1e-6)


def test_main_window_phase_spinbox_and_handle_stay_in_sync(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    window.phase1_spin.setValue(45.5)
    assert window.phase1_handle.value() == pytest.approx(45.5)


def test_main_window_pivot_defaults_to_spectrum_centre(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    index = window._selected_index()
    freq_ppm = window.session.spectra[index].axis_values(0, "ppm")
    expected_centre = (freq_ppm.min() + freq_ppm.max()) / 2
    # abs, not the default relative tolerance: pivot_spin displays 3
    # decimal places, so its value is rounded to 5e-4 regardless of
    # where the axis sits. The relative default only passed before
    # because this fixture's axis happened to be centred near -320 ppm,
    # where 1e-6 relative is larger than that rounding.
    assert window.pivot_spin.value() == pytest.approx(expected_centre, abs=5e-4)


def test_main_window_pivot_slider_change_compensates_phase0_so_data_does_not_jump(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    index = window._selected_index()
    window.phase1_spin.setValue(40.0)
    before = window.session.spectra[index].data.copy()

    window.pivot_slider.setValue(window.pivot_slider.value() + 50)

    after = window.session.spectra[index].data
    # atol loose enough to absorb phase0_spin's own 2-decimal-place
    # display rounding (the compensation is exact in floating point;
    # only the spin box's stored value is rounded) -- still far
    # tighter than the O(1) jump an uncompensated pivot move would
    # produce for a phase1 this size.
    np.testing.assert_allclose(before, after, atol=1e-3)
    assert window.phase0_spin.value() != 0.0


def test_main_window_pivot_change_with_zero_phase1_leaves_phase0_untouched(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    window.pivot_spin.setValue(window.pivot_spin.value() + 1.0)

    assert window.phase0_spin.value() == 0.0


def test_main_window_phase_adjustment_preserves_zoomed_view(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    # Simulate the user having zoomed in before touching the phase.
    window.canvas.ax.set_xlim(2.0, 4.0)
    window.canvas.ax.set_ylim(-0.5, 0.5)

    window.phase0_handle.setValue(45.0)

    assert window.canvas.ax.get_xlim() == (2.0, 4.0)
    assert window.canvas.ax.get_ylim() == (-0.5, 0.5)


def test_main_window_pivot_change_keeps_data_continuous(qapp, tmp_path):
    # Moving the pivot alone must not visibly change the spectrum --
    # Phase 0 is auto-compensated (see
    # _compensate_phase0_for_pivot_change) so the applied correction
    # stays exactly what it was; only where *future* Phase 1 drags
    # rotate around actually changes.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    index = window._selected_index()

    window.phase1_spin.setValue(60.0)
    with_default_pivot = window.session.spectra[index].data.copy()

    window.pivot_spin.setValue(window.pivot_spin.value() + 1.0)
    with_shifted_pivot = window.session.spectra[index].data

    # atol loose enough to absorb phase0_spin's own 2-decimal-place
    # display rounding -- see the matching comment in
    # test_main_window_pivot_slider_change_compensates_phase0_so_data_does_not_jump.
    np.testing.assert_allclose(with_default_pivot, with_shifted_pivot, atol=1e-3)
    assert window.phase0_spin.value() != pytest.approx(0.0)


def test_main_window_phase_range_is_wide(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.phase1_spin.setValue(1000.0)
    assert window.phase1_spin.value() == pytest.approx(1000.0)


def test_main_window_phase_range_is_genuinely_unbounded(qapp, tmp_path):
    # Previously capped at +/-36000 -- a joint-phased or heavily
    # multi-cycle-wrapped correction can genuinely exceed that.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    for spin in (window.phase0_spin, window.phase1_spin):
        spin.setValue(1_000_000.0)
        assert spin.value() == pytest.approx(1_000_000.0)
        spin.setValue(-1_000_000.0)
        assert spin.value() == pytest.approx(-1_000_000.0)


def test_main_window_phase1_step_is_larger_than_phase0(qapp, tmp_path):
    # Phase 1's per-notch/per-click increment must be much coarser than
    # Phase 0's -- a linear (whole-spectrum) correction needs a much
    # bigger raw-degree change than a zero-order one to visibly move
    # anything, per explicit user feedback.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.phase1_spin.singleStep() > window.phase0_spin.singleStep()
    assert window.phase1_handle.step > window.phase0_handle.step


def test_main_window_pivot_marker_appears_while_dragging_phase(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    assert window.canvas._pivot_line is None  # not shown before any phase interaction

    window.phase0_handle.setValue(45.0)

    assert window.canvas._pivot_line is not None
    assert window.canvas._pivot_line.get_xdata()[0] == pytest.approx(window.pivot_spin.value())

    window.pivot_spin.setValue(window.pivot_spin.value() + 1.0)
    assert window.canvas._pivot_line.get_xdata()[0] == pytest.approx(window.pivot_spin.value())


def test_main_window_quick_phase_buttons_nudge_from_current_value(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    window._on_phase0_nudge_clicked(90.0)
    assert window.phase0_spin.value() == pytest.approx(90.0)

    window._on_phase0_nudge_clicked(90.0)  # composes: two +90s == +180
    assert window.phase0_spin.value() == pytest.approx(180.0)

    window._on_phase0_nudge_clicked(-180.0)
    assert window.phase0_spin.value() == pytest.approx(0.0)
    assert window.phase0_handle.value() == pytest.approx(0.0)  # stays in sync


def test_main_window_zero_phase_buttons_zero_only_their_own_order(qapp, tmp_path):
    # Real user request: "there should be a button to zero 0th or 1st
    # order phase" -- distinct from the combined Reset button, which
    # zeros both together.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    window.phase0_spin.setValue(45.0)
    window.phase1_spin.setValue(30.0)

    window._on_zero_phase_clicked(0)
    assert window.phase0_spin.value() == pytest.approx(0.0)
    assert window.phase0_handle.value() == pytest.approx(0.0)
    assert window.phase1_spin.value() == pytest.approx(30.0)  # untouched

    window._on_zero_phase_clicked(1)
    assert window.phase1_spin.value() == pytest.approx(0.0)
    assert window.phase1_handle.value() == pytest.approx(0.0)


def test_main_window_committed_phase0_wraps_cyclically_past_360(qapp, tmp_path):
    # Real user request: "make 0th order phase cyclic from 0 to 359" --
    # phase0 is a full rotation (exp(i*phase0)), physically identical
    # 360deg away, so it shouldn't grow unbounded across many separate
    # drag sessions the way phase1 (a ramp rate, not a rotation) still
    # should.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    spectrum.metadata.raw["phase0"] = 0.0

    window.phase0_handle.setValue(400.0)  # deliberately past a full rotation
    window._capture_phase_reference()  # commits the pending delta
    assert spectrum.metadata.raw["phase0"] == pytest.approx(40.0)


def test_main_window_committed_phase0_wraps_negative_deltas_into_range(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    spectrum.metadata.raw["phase0"] = 40.0

    window.phase0_handle.setValue(-100.0)  # would go negative without wrapping
    window._capture_phase_reference()
    assert spectrum.metadata.raw["phase0"] == pytest.approx(300.0)


def test_main_window_phase1_is_not_wrapped(qapp, tmp_path):
    # Phase 1 is deliberately excluded -- a ramp rate across the
    # spectrum, not a rotation, so "modulo 360" has no equivalent
    # physical meaning for it (matches _PHASE_SPIN_RANGE's own
    # genuinely-unbounded design for the live delta itself).
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    spectrum.metadata.raw["phase1"] = 0.0

    window.phase1_handle.setValue(400.0)
    window._capture_phase_reference()
    assert spectrum.metadata.raw["phase1"] == pytest.approx(400.0)


def test_main_window_phase0_display_wraps_mid_drag_before_commit(qapp, tmp_path):
    # The display must stay cyclic continuously during an active drag,
    # not just once it's committed.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    spectrum.metadata.raw["phase0"] = 0.0
    window._capture_phase_reference()  # re-anchor against the baseline set above

    window.phase0_handle.setValue(370.0)  # not yet committed
    assert spectrum.metadata.raw["phase0"] == pytest.approx(0.0)  # unchanged until commit
    assert window.phase0_current_display.text() == "10.00°"  # but the display already wraps


def test_main_window_f1_phase0_also_wraps_cyclically(qapp, tmp_path, monkeypatch):
    # Same wrap must apply to phase0_f1 (the indirect dimension's own
    # zero-order phase), not just F2's plain "phase0" key.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[window._selected_index()]

    window.dim_combo.setCurrentIndex(1)  # F1 focused
    window.phase0_handle.setValue(400.0)
    window._capture_phase_reference()  # commits
    assert spectrum.metadata.raw["phase0_f1"] == pytest.approx(40.0)


def test_main_window_autophase_button_sets_phase0_and_resets_phase1(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    window.phase1_spin.setValue(15.0)
    window._on_autophase_clicked()  # must not raise

    assert window.phase1_spin.value() == pytest.approx(0.0)
    assert window.phase1_handle.value() == pytest.approx(0.0)
    assert window.phase0_handle.value() == pytest.approx(window.phase0_spin.value())


def test_main_window_autophase_button_noop_with_no_spectrum(qapp):
    window = MainWindow()
    window._on_autophase_clicked()  # must not raise


def test_main_window_autophase_stays_on_processing_preview_for_1d(qapp, tmp_path, monkeypatch):
    # Explicit user feedback: Autophase was pulling focus out of the
    # Processing-preview tab. A 1D spectrum's own phase update is
    # cheap regardless of which canvas shows it, so this should also
    # actually refresh the preview's content, not just avoid switching
    # away from it.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.phase0_spin.setValue(90.0)  # a deliberately wrong starting phase
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    before = spec_ax.lines[0].get_ydata().copy()

    window._on_autophase_clicked()  # should correct the phase back away from 90 deg

    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    after = spec_ax.lines[0].get_ydata()
    assert not np.allclose(before, after)


def test_main_window_autophase_stays_on_processing_preview_for_pseudo2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas

    window._on_autophase_clicked()  # must not raise, must not switch away

    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas


def test_main_window_revert_autophase_stays_on_processing_preview(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window._on_autophase_clicked()
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas

    window._on_revert_autophase_clicked()  # must not raise, must not switch away

    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas


def test_main_window_baseline_correct_stays_on_processing_preview_for_pseudo2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas

    window.baseline_correct_button.setChecked(True)  # must not raise, must not switch away

    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas


def test_main_window_phase_drag_tick_refreshes_processing_preview_live(qapp, tmp_path, monkeypatch):
    # Explicit user feedback: phasing wasn't updating the Processing-
    # preview tab at all (only the Spectrum tab) -- computed entirely
    # in memory now (see _compute_processing_preview, using the cached
    # raw FID), so it's cheap enough to refresh on every single drag
    # tick, not just once a gesture finishes.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    before = spec_ax.lines[0].get_ydata().copy()

    window.phase0_handle.setValue(45.0)  # one drag tick, no release

    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    after = spec_ax.lines[0].get_ydata()
    assert not np.allclose(before, after)


def test_main_window_phase_drag_preserves_processing_preview_zoom(qapp, tmp_path, monkeypatch):
    # Explicit user request: should be able to zoom into a small peak
    # and still phase it, without the view snapping back out on every
    # tick.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    spec_ax.set_xlim(4.0, 6.0)
    spec_ax.set_ylim(0.1, 0.2)

    window.phase0_handle.setValue(45.0)
    window.phase0_handle.setValue(90.0)  # a second tick, in case only the first preserved it

    assert spec_ax.get_xlim() == pytest.approx((4.0, 6.0))
    assert spec_ax.get_ylim() == pytest.approx((0.1, 0.2))


def test_main_window_phase_drag_shows_pivot_marker_on_processing_preview(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    window.phase0_handle.setValue(45.0)

    assert window.processing_preview_canvas._pivot_line is not None
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    assert window.processing_preview_canvas._pivot_marker_ax is spec_ax


def test_main_window_processing_preview_matches_spectrum_tab_phase(qapp, tmp_path, monkeypatch):
    # The bug as reported: phasing kept working on the Spectrum tab
    # but not on the Processing-preview tab -- assert the two actually
    # agree, not just that the preview changed at all.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window._set_zero_fill_value(64)  # isolate the phase check from the new default-zero-fill behaviour
    window.phase0_spin.setValue(45.0)
    spectrum_ydata = window.canvas.ax.lines[0].get_ydata().copy()

    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    preview_ydata = spec_ax.lines[0].get_ydata()

    np.testing.assert_allclose(preview_ydata, spectrum_ydata, atol=1e-6)


def test_main_window_apply_all_slices_checked_by_default(qapp, tmp_path, monkeypatch):
    # Explicit user feedback: a shared phase/baseline correction across
    # the whole pseudo-2D series is the common case -- unchecked by
    # default meant switching slices (each keeping its own, usually
    # untouched phase) looked like "processing resets" on every switch.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    assert window.apply_all_slices_checkbox.isChecked() is True


def test_main_window_phase_delta_commits_to_metadata_on_slice_switch(qapp, tmp_path, monkeypatch):
    # A live phase drag already writes the correction straight into
    # spectrum.data (verified elsewhere) -- but metadata.raw["phase0"]
    # is what _compute_processing_preview rebuilds from when it
    # recomputes from the raw FID, and nothing used to update it. That
    # gap meant the Processing tab's own preview looked un-phased again
    # after switching slices and back, even though spectrum.data itself
    # was never actually reset.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    baseline_phase0 = float(spectrum.metadata.raw.get("phase0", 0.0))

    window.phase0_spin.setValue(45.0)
    # Still pending -- not yet committed, since nothing re-anchored.
    assert spectrum.metadata.raw.get("phase0", 0.0) == pytest.approx(baseline_phase0)

    window.slice_spin.setValue(2)  # triggers _capture_phase_reference

    assert spectrum.metadata.raw["phase0"] == pytest.approx(baseline_phase0 + 45.0)
    assert window.phase0_spin.value() == pytest.approx(0.0)  # re-anchored, delta consumed


def test_main_window_per_slice_phase_delta_not_committed_to_metadata(qapp, tmp_path, monkeypatch):
    # A per-slice-only edit (apply-to-every-slice unchecked) can't be
    # expressed as one scalar in metadata.raw, which describes the
    # whole dataset -- _reapply_phase_post_reload already handles that
    # case's own persistence need directly, so metadata.raw must stay
    # untouched here rather than silently recording the wrong thing.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.apply_all_slices_checkbox.setChecked(False)
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    baseline_phase0 = float(spectrum.metadata.raw.get("phase0", 0.0))

    window.phase0_spin.setValue(45.0)
    window.slice_spin.setValue(2)

    assert spectrum.metadata.raw.get("phase0", 0.0) == pytest.approx(baseline_phase0)


def test_main_window_processing_preview_phase_survives_slice_switch(qapp, tmp_path, monkeypatch):
    # The actual end-to-end symptom: phase a pseudo-2D series (apply to
    # every slice, the default), switch away to a different slice and
    # back, then check the Processing tab's own preview -- computed
    # from the raw FID + metadata, not read off the live spectrum
    # directly -- still shows the phase actually applied.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=64), vendor="bruker")
    window._set_zero_fill_value(64)  # isolate the phase check from zero-fill shape changes
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    # This is about phase persistence across a slice switch, not
    # baseline -- disable the now-on-by-default baseline correction so
    # its own independent sigma-clip fit (run separately for the
    # preview and the live spectrum) can't introduce an unrelated
    # mismatch between the two.
    window.baseline_correct_button.setChecked(False)

    window.phase0_spin.setValue(90.0)
    window.slice_spin.setValue(2)
    window.slice_spin.setValue(1)

    _fid, _dt, preview_spectrum, _title = window._compute_processing_preview(spectrum)
    np.testing.assert_allclose(preview_spectrum.data, spectrum.data[0], atol=1e-6)


def test_main_window_processing_preview_normalizes_by_ns(qapp, tmp_path, monkeypatch):
    # Real bug, found on real data (NS=2600): _compute_processing_
    # preview built its preview straight from the raw FID without ever
    # dividing by NS (scans), unlike read_1d/read_2d's own load path
    # (data = spectrum / mass / ns) -- looked like "phase isn't applied
    # the same way in the Spectrum and Processing tabs" (actually an
    # amplitude-scale mismatch of NS-to-1, not a phase bug at all,
    # since every real acquisition has NS > 1). The previous test
    # (`..._phase_survives_slice_switch`) didn't catch this because its
    # fixture's default NS=1.0 makes the missing division a no-op.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=64, ns=8.0), vendor="bruker")
    window._set_zero_fill_value(64)
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    window.baseline_correct_button.setChecked(False)  # isolate the NS check from baseline correction

    _fid, _dt, preview_spectrum, _title = window._compute_processing_preview(spectrum)

    np.testing.assert_allclose(preview_spectrum.data, spectrum.data[0], atol=1e-6)


def test_main_window_1d_phase_drag_uses_same_source_as_processing_preview(qapp, tmp_path, monkeypatch):
    # Regression test: dragging Phase 0/1 for a 1D spectrum used to
    # apply the correction directly to a snapshot taken back when the
    # Phase controls were last anchored (_phase_reference_full) --
    # completely independent of whatever window/zero-fill choice the
    # Processing tab's own live preview (_compute_processing_preview)
    # was *currently* showing. If that choice had changed since
    # anchoring but not yet been "Applied", the Spectrum tab and the
    # Processing tab's preview could show different point counts
    # entirely, not just a numeric mismatch -- explicit user report:
    # "a bug with phasing 1d from the processing to the spectrum tab...
    # spectrum should be populated from the processing window data and
    # processing metadata." Fixed: the "1d" phase-drag branch now
    # calls _compute_processing_preview itself and writes its result
    # straight into spectrum.data/data_imag/axes -- the same source,
    # not a second computation of it.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)  # live on the Processing tab, not yet "Applied"
    window._set_zero_fill_value(256)  # also live-only -- as-loaded spectrum is 64 points

    window.phase0_spin.setValue(25.0)

    spectrum = window.session.spectra[index]
    assert len(spectrum.data) == 256  # picked up the live zero-fill, not stuck at the stale as-loaded length

    computed = window._compute_processing_preview(spectrum)
    _fid, _dt, preview_spectrum, _title = computed
    np.testing.assert_allclose(spectrum.data, preview_spectrum.data, atol=1e-10)
    np.testing.assert_allclose(spectrum.data_imag, preview_spectrum.data_imag, atol=1e-10)


def test_main_window_1d_phase_drag_updates_spectrum_tab_canvas_on_length_change(qapp, tmp_path, monkeypatch):
    # Companion to the "same source" test above, from the canvas side:
    # MplCanvas.update_data assumed a phase-only change never touches
    # length, so it kept reusing whatever x-positions were already on
    # screen -- now that a phase drag can also pick up a live zero-fill
    # change, it must fall back to a full rebuild instead of plotting
    # new y-values against stale, wrong-length x-data.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    window._set_zero_fill_value(256)
    window.phase0_spin.setValue(25.0)  # must not raise, and must actually rebuild the line

    assert len(window.canvas.ax.lines[0].get_ydata()) == 256


def test_main_window_pivot_slider_moves_pivot_spin(qapp, tmp_path):
    # The slider is inverted relative to raw ppm value, to match the
    # spectrum's own inverted x-axis (high ppm on the left) -- dragging
    # the slider right should move the pivot marker right on screen,
    # which is toward LOW ppm, not high.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    freq_ppm = window._phase_freq_ppm

    # abs tolerance -- pivot_spin rounds to 3 decimal places (see
    # test_main_window_pivot_defaults_to_spectrum_centre).
    window.pivot_slider.setValue(0)  # left end of the slider
    assert window.pivot_spin.value() == pytest.approx(float(freq_ppm.max()), abs=5e-4)

    window.pivot_slider.setValue(1000)  # right end of the slider
    assert window.pivot_spin.value() == pytest.approx(float(freq_ppm.min()), abs=5e-4)


def test_main_window_pivot_spin_moves_pivot_slider(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    freq_ppm = window._phase_freq_ppm

    window.pivot_spin.setValue(float(freq_ppm.max()))
    assert window.pivot_slider.value() == 0

    window.pivot_spin.setValue(float(freq_ppm.min()))
    assert window.pivot_slider.value() == 1000


def test_main_window_pivot_change_after_live_zero_fill_does_not_crash(qapp, tmp_path, monkeypatch):
    # Real user-reported crash: ValueError: x and y must have same
    # first dimension (8192 vs 4096), dragging the phase pivot slider
    # on a pseudo-2D slice. Root cause: _phase_reference_full/
    # _phase_freq_ppm are a one-time snapshot (_capture_phase_reference,
    # "never called mid-drag") -- a live FID-processing change (e.g.
    # zero-fill) that changes spectrum.data/axes' own shape via the
    # debounced _recompute_multi_row_data left that snapshot stale, so
    # the next phase/pivot interaction wrote the *old*-shaped snapshot
    # straight into spectrum.data while spectrum.axes had already
    # moved to the *new* shape. Reproduces the exact real sequence:
    # load, change zero-fill live, let the debounce settle, then move
    # the pivot slider -- must not raise.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=32, f1_nuc="off"), vendor="bruker")
    index = window._selected_index()
    assert window._phase_mode == "pseudo2d"
    original_points = window.session.spectra[index].data.shape[1]

    window._set_zero_fill_value(original_points * 2)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)
    assert window.session.spectra[index].data.shape[1] == original_points * 2

    window._on_pivot_slider_changed(500)  # must not raise

    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[1] == len(spectrum.axes[-1].ppm)


def test_main_window_moving_pivot_does_not_reset_phase(qapp, tmp_path):
    # Explicit user requirement: moving the pivot (spin or slider)
    # must never zero out an in-progress Phase 0/Phase 1 correction --
    # only _capture_phase_reference (reselection, not a pivot move)
    # should ever do that. Phase 0's own *number* is expected to
    # change to compensate (see _compensate_phase0_for_pivot_change),
    # keeping the applied correction itself continuous rather than its
    # raw value fixed -- Phase 1 is untouched either way.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    window.phase0_spin.setValue(30.0)
    window.phase1_spin.setValue(-10.0)

    window.pivot_slider.setValue(750)
    assert window.phase0_spin.value() != pytest.approx(0.0)
    assert window.phase1_spin.value() == pytest.approx(-10.0)

    window.pivot_spin.setValue(window.pivot_spin.value() + 1.0)
    assert window.phase0_spin.value() != pytest.approx(0.0)
    assert window.phase1_spin.value() == pytest.approx(-10.0)


def test_main_window_autophase_joint_uses_autophase01(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    window.autophase_joint_checkbox.setChecked(True)

    from opennmr.gui import main_window as mw
    monkeypatch.setattr(mw.processing_ft, "autophase01", lambda *a, **kw: (12.0, 7.0))

    window._on_autophase_clicked()

    assert window.phase0_spin.value() == pytest.approx(12.0)
    assert window.phase1_spin.value() == pytest.approx(7.0)


def test_main_window_revert_autophase_button_disabled_until_autophase(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    assert window.revert_autophase_button.isEnabled() is False

    window._on_autophase_clicked()
    assert window.revert_autophase_button.isEnabled() is True


def test_main_window_revert_autophase_restores_previous_phase(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    window.phase0_spin.setValue(42.0)
    window.phase1_spin.setValue(-5.0)
    window._on_autophase_clicked()
    assert window.phase0_spin.value() != pytest.approx(42.0)  # autophase actually changed it

    window._on_revert_autophase_clicked()

    assert window.phase0_spin.value() == pytest.approx(42.0)
    assert window.phase1_spin.value() == pytest.approx(-5.0)
    assert window.revert_autophase_button.isEnabled() is False


def test_main_window_revert_autophase_noop_with_nothing_to_revert(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    window._on_revert_autophase_clicked()  # must not raise


def test_main_window_per_slice_phases_survive_a_reload(qapp, tmp_path, monkeypatch):
    # Reported directly: "need to make sure that per-slice phasing/
    # processing is being maintained across loads". They were saved and
    # restored into metadata.raw correctly -- and never *applied*,
    # because the reader has no parameter for them and the load took
    # the already-applied fast path anyway.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=6, n_f2=64)
    window._load(path, vendor="bruker")
    spectrum = window.session.spectra[0]

    for row in range(spectrum.data.shape[0]):
        window._add_slice_phase(spectrum, row, 10.0 * (row + 1), 0.0)
    window._recompute_multi_row_data(spectrum)
    phased = spectrum.complex_data.copy()
    window._schedule_sidecar_save(0)
    window._sidecar_save_timer.stop()
    window._run_debounced_sidecar_save()

    reloaded_window = MainWindow()
    reloaded_window._load(path, vendor="bruker")
    reloaded = reloaded_window.session.spectra[0]

    assert reloaded.metadata.raw["slice_phase0"] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # Recorded *and* applied -- the data itself comes back phased.
    np.testing.assert_allclose(reloaded.complex_data, phased)


def test_main_window_a_series_with_no_slice_phases_keeps_the_fast_load(qapp, tmp_path, monkeypatch):
    # The exclusion must be conditional: falling back to the full
    # recompute for every 2D would undo the load-speed work.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=6, n_f2=64)
    window._load(path, vendor="bruker")
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["_read_used_saved_processing"] = True
    spectrum.metadata.raw["slice_phase0"] = [0.0] * spectrum.data.shape[0]

    assert window._read_already_applied_saved_processing(spectrum) is True

    spectrum.metadata.raw["slice_phase0"][2] = 15.0
    assert window._read_already_applied_saved_processing(spectrum) is False


def test_main_window_baseline_correct_is_off_by_default(qapp, tmp_path):
    # **This default has been both ways.** It was turned *on* for
    # earlier feedback that "a well-phased spectrum should generally
    # land close to zero"; it is now off again, chosen deliberately
    # when the baseline overlay landed -- a spectrum that arrives
    # already corrected leaves the overlay nothing meaningful to draw,
    # since the trace it would fit is the one the correction has
    # already flattened.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    assert window.baseline_correct_button.isChecked() is False
    assert window.session.spectra[0].metadata.raw.get("baseline_correct") is False


def test_main_window_baseline_correct_changes_1d_data(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    index = window._selected_index()
    window.baseline_correct_button.setChecked(False)  # start from the uncorrected state
    original = window.session.spectra[index].data.copy()

    window.baseline_correct_button.setChecked(True)

    assert not np.allclose(original, window.session.spectra[index].data)


def test_main_window_baseline_correct_is_a_toggle_not_one_shot(qapp, tmp_path):
    # Explicit user feedback ("baseline correction should be applied
    # to the processed spectrum but the correction should be able to
    # be switched on and off") -- unchecking must restore exactly what
    # was there before, not just leave whatever the last correction
    # produced.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    index = window._selected_index()
    window.baseline_correct_button.setChecked(False)  # start from the uncorrected state
    original = window.session.spectra[index].data.copy()

    window.baseline_correct_button.setChecked(True)
    assert not np.allclose(original, window.session.spectra[index].data)

    window.baseline_correct_button.setChecked(False)
    np.testing.assert_allclose(window.session.spectra[index].data, original)


def test_main_window_baseline_correct_pseudo2d_respects_apply_all_slices(qapp):
    ppm = np.linspace(-10.0, 10.0, 300)
    n_rows = 3
    rows = [0.02 * (ppm - 3.0) ** 2 + i + 5.0 * np.exp(-((ppm - 1.0) ** 2) / (2 * 0.3 ** 2)) for i in range(n_rows)]
    data = np.array(rows)
    axes = [NMRAxis(kind="index", index=np.arange(1, n_rows + 1)), NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)]
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    spectrum = NMRSpectrum(data=data, axes=axes, metadata=metadata, data_imag=np.zeros_like(data))

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window.apply_all_slices_checkbox.isChecked() is True  # established default

    original = window.session.spectra[0].data.copy()
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("spline"))
    window.baseline_correct_button.setChecked(True)
    _settle_live_reprocess(window)  # "apply to every slice" debounces the actual recompute
    corrected = window.session.spectra[0].data
    for row in range(n_rows):
        assert not np.allclose(original[row], corrected[row])

    # Single-slice targeting only touches the selected row.
    window.baseline_correct_button.setChecked(False)
    window.apply_all_slices_checkbox.setChecked(False)
    window.slice_spin.setValue(2)
    before = window.session.spectra[0].data.copy()
    window.baseline_correct_button.setChecked(True)
    after = window.session.spectra[0].data
    assert not np.allclose(before[1], after[1])
    assert np.allclose(before[0], after[0])
    assert np.allclose(before[2], after[2])


def test_main_window_baseline_correct_conventional_2d(qapp):
    ppm = np.linspace(-10.0, 10.0, 150)
    n_f1 = 3
    rows = [0.02 * (ppm - 3.0) ** 2 + i + 5.0 * np.exp(-((ppm - 1.0) ** 2) / (2 * 0.3 ** 2)) for i in range(n_f1)]
    data = np.array(rows)
    axes = [
        NMRAxis(kind="frequency", ppm=np.linspace(0, 1, n_f1), hz=np.linspace(0, 100, n_f1)),
        NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0),
    ]
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=False)
    spectrum = NMRSpectrum(data=data, axes=axes, metadata=metadata, data_imag=np.zeros_like(data))

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window._phase_mode == "2d"

    original = window.session.spectra[0].data.copy()
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("spline"))
    window.baseline_correct_button.setChecked(True)
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute
    corrected = window.session.spectra[0].data
    for row in range(n_f1):
        assert not np.allclose(original[row], corrected[row])


def test_main_window_baseline_correct_spline_method_persists_across_reselection(qapp, tmp_path):
    # Explicit user feedback asking for "options to spline... or
    # something" (linear systematically under-corrects a curved
    # baseline), plus the broader "each experiment object should have
    # the metadata locked in for when it is called upon" ask -- the
    # toggle/method choice must survive switching away and back.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64, name="expt_a"), vendor="bruker")
    index_a = window._selected_index()
    window._load(_make_1d_dataset(tmp_path, n=64, name="expt_b"), vendor="bruker")
    index_b = window._selected_index()

    window._select_single_row(index_a)
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("spline"))
    window.baseline_correct_button.setChecked(True)

    window._select_single_row(index_b)
    assert window.baseline_correct_button.isChecked() is False  # untouched spectrum, own (off) default
    assert window.baseline_method_combo.currentData() == "linear"  # its own default method, not expt_a's spline

    window._select_single_row(index_a)
    assert window.baseline_correct_button.isChecked() is True
    assert window.baseline_method_combo.currentData() == "spline"


def test_main_window_baseline_degree_spin_only_visible_for_polynomial(qapp):
    window = MainWindow()
    assert window.baseline_method_combo.currentData() == "linear"
    assert window.baseline_degree_spin.isHidden() is True

    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("polynomial"))
    assert window.baseline_degree_spin.isHidden() is False
    assert window.baseline_smoothing_spin.isHidden() is True

    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("spline"))
    assert window.baseline_degree_spin.isHidden() is True
    assert window.baseline_smoothing_spin.isHidden() is False


def test_main_window_baseline_correct_polynomial_degree_persists_across_reselection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64, name="expt_a"), vendor="bruker")
    index_a = window._selected_index()
    window._load(_make_1d_dataset(tmp_path, n=64, name="expt_b"), vendor="bruker")
    index_b = window._selected_index()

    window._select_single_row(index_a)
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("polynomial"))
    window.baseline_degree_spin.setValue(4)
    window.baseline_correct_button.setChecked(True)

    window._select_single_row(index_b)
    assert window.baseline_degree_spin.value() == 2  # untouched spectrum, own default

    window._select_single_row(index_a)
    assert window.baseline_method_combo.currentData() == "polynomial"
    assert window.baseline_degree_spin.value() == 4


def test_main_window_baseline_correct_polynomial_and_sinusoidal_change_the_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64, name="expt"), vendor="bruker")
    index = window._selected_index()

    for method in ("polynomial", "sinusoidal"):
        window.baseline_correct_button.setChecked(False)
        window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData(method))
        original = window.session.spectra[index].data.copy()
        window.baseline_correct_button.setChecked(True)
        corrected = window.session.spectra[index].data
        assert not np.allclose(original, corrected)


def test_main_window_baseline_correct_noop_with_no_spectrum(qapp):
    window = MainWindow()
    window.baseline_correct_button.setChecked(True)  # must not raise


def test_main_window_load_failure_shows_error_not_crash(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = {}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))

    window = MainWindow()
    window._load(str(tmp_path / "does_not_exist"), vendor="bruker")
    assert shown.get("called") is True
    assert window.spectrum_list.count() == 0


# ============================================================
# 2D / pseudo-2D phase processing
# ============================================================

def _complex(spectrum):
    return spectrum.data + 1j * spectrum.data_imag


def test_main_window_processing_enabled_for_2d_with_imag(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, with_imag=True), vendor="bruker")
    assert window.phase0_handle.isEnabled() is True


def test_main_window_processing_disabled_for_2d_without_imag(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, with_imag=False), vendor="bruker")
    assert window.phase0_handle.isEnabled() is False
    assert window.dimension_box.isHidden() is True


def test_main_window_dimension_box_shows_dim_combo_for_conventional_2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt out of the pseudo-2D default

    assert window.dimension_box.isHidden() is False
    assert window.dim_combo.isHidden() is False
    assert window.slice_spin.isHidden() is True


def test_main_window_dimension_box_shows_slice_controls_for_pseudo_2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    # Pseudo-2D is already the default straight off _load -- no toggle needed.

    assert window.dimension_box.isHidden() is False
    assert window.slice_spin.isHidden() is False
    assert window.dim_combo.isHidden() is True
    assert window.slice_spin.maximum() == 4


def test_main_window_2d_f2_phase_applies_uniformly_across_rows(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=16), vendor="bruker")
    index = window._selected_index()
    original = _complex(window.session.spectra[index]).copy()

    window.phase0_handle.setValue(45.0)  # dim_combo defaults to Direct (F2)

    updated = _complex(window.session.spectra[index])
    assert not np.allclose(original, updated)
    # A pure phase correction never changes magnitude -- true regardless
    # of which dimension/rows it was broadcast across.
    np.testing.assert_allclose(np.abs(updated), np.abs(original), atol=1e-6)


def test_main_window_2d_f1_phase_applies_uniformly_across_columns(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=8, n_f2=4), vendor="bruker")
    index = window._selected_index()
    original = _complex(window.session.spectra[index]).copy()

    window.dim_combo.setCurrentIndex(1)  # Indirect (F1)
    window.phase0_handle.setValue(45.0)

    updated = _complex(window.session.spectra[index])
    assert not np.allclose(original, updated)
    np.testing.assert_allclose(np.abs(updated), np.abs(original), atol=1e-6)


def test_main_window_2d_dimension_toggle_resets_phase_controls(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.phase0_handle.setValue(30.0)

    window.dim_combo.setCurrentIndex(1)

    assert window.phase0_spin.value() == pytest.approx(0.0)
    assert window.phase0_handle.value() == pytest.approx(0.0)


def test_main_window_phase_current_display_is_read_only(qapp):
    # Explicit user request: "just display non-editable numbers with
    # current phase instead of that editable box" -- phase0_spin/
    # phase1_spin still exist (see _make_phase_control's own comment)
    # but are never added to any layout; these read-only labels are
    # what the user actually sees and interacts with (via the drag
    # handle, not by typing into the display itself).
    window = MainWindow()
    for display in (window.phase0_current_display, window.phase1_current_display):
        assert isinstance(display, QLabel)
        assert display.parent() is not None  # actually placed in the layout
    assert window.phase0_spin.parent() is None  # the old editable control is not


def test_main_window_current_phase_display_shows_dash_with_nothing_selected(qapp):
    window = MainWindow()
    assert window.phase0_current_display.text() == "—"
    assert window.phase1_current_display.text() == "—"


def test_main_window_current_phase_display_updates_live_during_a_drag(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.phase0_current_display.text() == "0.00°"
    assert window.phase1_current_display.text() == "0.00°"

    window.phase0_handle.setValue(45.0)

    assert window.phase0_current_display.text() == "45.00°"
    assert window.phase1_current_display.text() == "0.00°"
    assert window.phase0_spin.value() == pytest.approx(45.0)  # the hidden delta control is untouched by this


def test_main_window_current_phase_display_survives_reselection_unlike_the_hidden_delta(qapp, tmp_path):
    # Explicit user feedback: Phase 0/Phase 1 resetting to 0 on
    # reselection read as the phase value itself being lost -- this
    # display is what actually shows the still-applied total, while
    # the hidden delta control keeps its own (deliberate, delta-from-
    # here) behavior unchanged.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="expt_a"), vendor="bruker")
    index_a = window._selected_index()
    window._load(_make_1d_dataset(tmp_path, name="expt_b"), vendor="bruker")

    window._select_single_row(index_a)
    window.phase0_handle.setValue(45.0)
    assert window.phase0_current_display.text() == "45.00°"

    window._select_single_row(1)  # away to B
    assert window.phase0_current_display.text() == "0.00°"  # B was never phased

    window._select_single_row(index_a)  # back to A
    assert window.phase0_spin.value() == pytest.approx(0.0)  # the hidden delta control resets, by design
    assert window.phase0_current_display.text() == "45.00°"  # but the real value is still shown


def test_main_window_2d_phase_defaults_to_slice_1(qapp, tmp_path):
    # Real user requirement: "the phasing for a 2D should still default
    # to a 1D slice like it does before you process the indirect
    # dimension rather than trying to phase the entire thing all at
    # once." Originally picked the largest-total-magnitude row/column
    # (needing the *processed* F1 axis to even evaluate); replaced per
    # explicit follow-up feedback ("the processing should just always
    # use slice 1 of the 'xf2' (unprocessed indirect)") with a fixed
    # index 1 -- always a valid position in both the raw and the
    # processed array, sidestepping the raw-vs-processed row-count
    # mismatch class of bug entirely.
    n_f1, n_f2 = 4, 8
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=n_f1, n_f2=n_f2), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    assert window._phase_mode == "2d"
    assert window._phase_2d_slice_index == 1


def test_main_window_2d_phase_stays_a_contour_not_a_slice(qapp, tmp_path, monkeypatch):
    # Real, repeated user requirement, superseding an earlier round's
    # opposite request: "the Spectrum tab should always show the 2D or
    # pseudo 2D where there is one" -- phasing a genuine 2D must never
    # switch self.canvas to a single-slice line the way it used to
    # (renamed from this test's own former name,
    # "..._shows_a_slice_not_a_full_contour", which asserted exactly
    # the behaviour this now forbids). _make_raw_2d_dataset, not
    # _make_2d_dataset -- the Processing tab's own auto-switch-to-
    # preview (_on_side_tab_changed) is gated on _fid_reprocessable,
    # which the plain (no cached raw FID) fixture never satisfies, so
    # the Processing-preview half of this test wouldn't exercise
    # anything real with it (confirmed directly: canvas_tabs never
    # actually switches tabs in that case at all).
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    assert window.canvas._contour_spectrum is not None  # starts as a contour

    window.phase0_handle.setValue(45.0)
    _settle_canvas_2d_refresh(window)

    assert window.canvas._contour_spectrum is not None  # still a contour
    assert len(window.canvas.ax.collections) >= 1

    # The 1D slice for judging phase now lives only on the Processing
    # tab's own preview -- "the processing tab should only have a 1D
    # that is matched to the fid tab."
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.processing_preview_canvas._contour_spectrum is None
    assert len(window.processing_preview_canvas.ax.lines) >= 1


def test_main_window_2d_phase_never_needs_reprocessing_process_indirect_to_recover_contour(
        qapp, tmp_path, monkeypatch):
    # Real user report, verbatim: "I still have to repeatedly select
    # and deselect the process indirect button to get back the 2D."
    # A real, multi-step sequence of phase-adjacent interactions --
    # drag, dimension switch, tab switch, per-slice autophase -- with
    # process_indirect_button *never* touched after the initial opt-in,
    # confirming self.canvas is a contour at every single point along
    # the way, not just immediately after one specific action.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)

    def assert_still_a_contour(label):
        assert window.canvas._contour_spectrum is not None, f"not a contour after: {label}"
        assert len(window.canvas.ax.collections) >= 1, f"no collections after: {label}"

    assert_still_a_contour("initial opt-in")

    window.phase0_handle.setValue(30.0)
    _settle_canvas_2d_refresh(window)
    assert_still_a_contour("phase0 drag + settle")

    window.dim_combo.setCurrentIndex(1)  # switch to F1 (indirect)
    _settle_canvas_2d_refresh(window)
    assert_still_a_contour("dimension switch")

    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert_still_a_contour("entering the Processing tab")

    window.phase1_handle.setValue(15.0)
    _settle_canvas_2d_refresh(window)
    assert_still_a_contour("phase1 drag while on the Processing tab")

    window.side_tabs.setCurrentIndex(0)  # back to the Spectra tab
    assert_still_a_contour("leaving the Processing tab")

    assert window.process_indirect_button.isChecked() is True  # never touched, still on


def test_main_window_2d_phase_still_applies_uniformly_to_every_row(qapp, tmp_path):
    # The *display* changed (see the test above), but the correction
    # itself must still be the one shared phase0/phase1 across the
    # whole array -- not silently narrowed to only the displayed row.
    n_f1, n_f2 = 4, 8
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=n_f1, n_f2=n_f2), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    index = window._selected_index()

    window.phase0_handle.setValue(90.0)

    spectrum = window.session.spectra[index]
    complex_data = spectrum.data + 1j * spectrum.data_imag
    reference = window._phase_reference_full
    freq_ppm = window._phase_freq_ppm
    for row in range(n_f1):
        expected = processing_ft.apply_phase(reference[row], freq_ppm, 90.0, 0.0, pivot_frac=0.5)
        np.testing.assert_allclose(complex_data[row], expected, atol=1e-8)


def test_main_window_dimension_switch_commits_pending_phase_to_the_correct_key(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: dim_combo.currentIndexChanged fires
    # *after* Qt has already updated dim_combo.currentIndex() to the
    # new value, so a stale read inside _commit_pending_phase_to_metadata
    # (itself triggered by that same signal, via _capture_phase_reference)
    # used to commit a pending F2 delta under the F1 key (or vice versa)
    # on every dimension switch -- confirmed directly against real 2D
    # data before this fix existed.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[window._selected_index()]
    baseline_phase0 = float(spectrum.metadata.raw.get("phase0", 0.0))

    assert window.dim_combo.currentIndex() == 0  # F2 (direct) focused
    window.phase0_handle.setValue(37.0)  # a pending F2 delta, not yet committed

    window.dim_combo.setCurrentIndex(1)  # switch focus to F1 mid-pending-delta

    assert spectrum.metadata.raw.get("phase0") == pytest.approx(baseline_phase0 + 37.0)
    assert spectrum.metadata.raw.get("phase0_f1") is None  # not the wrong key


def test_main_window_dimension_switch_commits_pending_phase_to_the_correct_key_reverse(
        qapp, tmp_path, monkeypatch):
    # Same bug, opposite direction (F1 -> F2), for symmetry.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[window._selected_index()]

    window.dim_combo.setCurrentIndex(1)  # F1 focused first
    window.phase0_handle.setValue(15.0)  # a pending F1 delta, not yet committed

    window.dim_combo.setCurrentIndex(0)  # switch focus to F2 mid-pending-delta

    assert spectrum.metadata.raw.get("phase0_f1") == pytest.approx(15.0)


def test_main_window_process_indirect_toggle_preserves_both_dimensions_phase(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: _current_phase_kwargs/_reapply_phase_
    # post_reload only ever preserved whichever dimension the Phase
    # panel's own Dimension combo happened to be currently focused on,
    # silently discarding the *other* dimension's already-established
    # phase on every "Process Indirect Dimension" toggle -- confirmed
    # directly against real 2D data before this fix existed. Explicit
    # user requirement: "phasing of 2D data before and after processing
    # the indirect dimension should use the same info."
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[window._selected_index()]

    window.phase0_handle.setValue(37.0)
    window.dim_combo.setCurrentIndex(1)  # commits the F2 delta, focus -> F1
    window.phase0_handle.setValue(15.0)
    window.dim_combo.setCurrentIndex(0)  # commits the F1 delta, focus -> F2 (left here)

    expected_phase0 = spectrum.metadata.raw.get("phase0")
    expected_phase0_f1 = spectrum.metadata.raw.get("phase0_f1")
    assert expected_phase0_f1 == pytest.approx(15.0)

    # F2 is focused right now -- toggling should still preserve F1's
    # own phase too, not just F2's (the bug: only the focused one did).
    window.process_indirect_button.setChecked(False)  # -> pseudo-2D
    window.process_indirect_button.setChecked(True)  # -> genuine 2D again

    new_spectrum = window.session.spectra[window._selected_index()]
    assert new_spectrum.metadata.raw.get("phase0") == pytest.approx(expected_phase0)
    assert new_spectrum.metadata.raw.get("phase0_f1") == pytest.approx(expected_phase0_f1)


def test_main_window_process_indirect_toggle_round_trip_preserves_f1_phase(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: even after the single-toggle
    # preservation fix above, F1's phase was only ever baked into
    # spectrum.data post-hoc -- never written back into the freshly-
    # reloaded spectrum's own metadata.raw -- so it survived exactly
    # one toggle and was then gone for good: a genuine 2D -> pseudo-2D
    # -> genuine 2D round trip (e.g. briefly checking the pseudo-2D
    # relaxation view and coming back) silently lost it, even though a
    # single toggle in isolation didn't. Checked across two round
    # trips, not just one, to rule out a one-shot fluke.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[window._selected_index()]

    window.dim_combo.setCurrentIndex(1)
    window.phase0_handle.setValue(15.0)
    window.dim_combo.setCurrentIndex(0)
    expected_phase0_f1 = spectrum.metadata.raw.get("phase0_f1")
    assert expected_phase0_f1 == pytest.approx(15.0)

    for round_trip in (1, 2):
        window.process_indirect_button.setChecked(False)
        window.process_indirect_button.setChecked(True)
        new_spectrum = window.session.spectra[window._selected_index()]
        assert new_spectrum.metadata.raw.get("phase0_f1") == pytest.approx(expected_phase0_f1), \
            f"F1 phase lost after round trip #{round_trip}"


def test_main_window_2d_phase_slice_follows_dimension_switch(qapp, tmp_path, monkeypatch):
    # _phase_2d_slice_index is now a fixed 1 regardless of dimension
    # (see test_main_window_2d_phase_defaults_to_slice_1), so the index
    # itself can no longer discriminate whether a dimension switch
    # actually re-anchors the phase-preview display -- checked via the
    # Processing-preview canvas's own title instead (the only place a
    # titled slice shows any more: self.canvas always stays a contour,
    # see test_main_window_2d_phase_stays_a_contour_not_a_slice).
    # _make_raw_2d_dataset -- see that same test's own comment on why
    # the plain fixture can't exercise the Processing-preview canvas
    # at all (_on_side_tab_changed's own _fid_reprocessable gate).
    n_f1, n_f2 = 4, 8
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=n_f1, n_f2=n_f2, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window._phase_2d_slice_index == 1  # F2 (direct) default: row 1

    window.dim_combo.setCurrentIndex(1)  # switch to F1 (indirect)

    assert window._phase_2d_slice_index == 1  # F1 default: column 1
    # The F1 view is a dual panel now (interferogram above spectrum),
    # so the title that proves the switch re-anchored the display is
    # the FID panel's. The interferogram deliberately shows the
    # *strongest* F2 column rather than column 1 -- the panel exists to
    # judge F1 truncation against a real decay, and column 1 is the
    # edge of the spectrum, which is noise.
    assert "F1 interferogram" in window.processing_preview_canvas._processing_preview_fid_ax.get_title()


def test_main_window_processing_preview_slice_matches_spectrum_tab_across_zero_fill_mismatch(
        qapp, tmp_path, monkeypatch):
    # Real user report: "there is something happening still with
    # phasing for 2D data, especially when switching between tabs."
    # _compute_processing_preview (the Processing tab's own live
    # preview) picked its row from slice_spin.value() -- always 1,
    # since that control is hidden for a genuine 2D -- instead of
    # _phase_2d_slice_index, the row actually shown on the Spectrum
    # tab while 2D-phasing. A first fix used _phase_2d_slice_index
    # directly as a raw_fid row index, which is *also* wrong:
    # _phase_2d_slice_index indexes the *processed*, zero-filled F1
    # axis (spectrum.data.shape[0]), while raw_fid
    # (metadata.raw["fid_raw"]) is the pre-zero-fill acquired array --
    # the two have different lengths whenever F1 zero-fill changes the
    # processed size, with no exact index correspondence between them
    # (a zero-filled row isn't a real acquired point). This test
    # deliberately zero-fills F1 from 4 raw rows to 16 processed rows
    # so the three candidate behaviours give three different, known
    # answers: the dead slice_spin path gives "Slice 1"; a direct/
    # clamped use of _phase_2d_slice_index saturates at the raw
    # array's own last row, "Slice 4"; only the proportional mapping
    # (_phase_2d_proportional_row) gives the correct "Slice 3" for
    # _phase_2d_slice_index == 9 of 16 processed rows against 4 raw
    # rows (fraction 8/15 -> round(1.6) -> raw row 2, 1-based 3).
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D
    window.dim_combo.setCurrentIndex(0)  # F2 (direct) -- the branch under test
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    window.zero_fill_f1_spin.setValue(16)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)
    assert spectrum.data.shape[0] == 16  # processed F1 now mismatches the 4 raw rows

    raw_fid = spectrum.metadata.raw["fid_raw"]
    assert raw_fid.shape[0] == 4  # the raw cache is untouched by the live F1 zero-fill

    window._phase_2d_slice_index = 9  # deliberately not proportional to either row count

    _fid, _dt, _preview_spectrum, title = window._compute_processing_preview(spectrum)

    assert title == "Slice 3"


def test_main_window_2d_genuine_processing_preview_shows_a_slice_not_a_contour(qapp, tmp_path, monkeypatch):
    # Real user report: "I get a different contour in processing vs
    # the spectrum/import tabs." Just landing on the Processing tab
    # for a genuine 2D (no phase-dragging involved at all)
    # used to fall back to a full contour redraw of the whole spectrum
    # (_show_processing_preview's own old "conventional 2D... falls
    # back to mirroring the Spectrum tab" branch) instead of a single
    # slice, unlike pseudo-2D, which already only ever showed one
    # slice here. Explicit follow-up ask: "the processing is just for
    # a single slice (first or largest magnitude)... This is what I
    # have been asking for for all 2D data."
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D

    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    canvas = window.processing_preview_canvas
    assert canvas._contour_spectrum is None  # a slice, not a contour
    assert len(canvas.ax.lines) >= 1


def test_main_window_2d_genuine_processing_preview_supports_f1_dimension(qapp, tmp_path, monkeypatch):
    # "should be independently possible for the indirect dimension" --
    # explicit follow-up ask alongside the single-slice fix above: F1
    # (indirect) previously fell back to the same full-contour view
    # regardless of dim_combo, since only F2 (direct) had a single-
    # slice preview implemented at all.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    window.dim_combo.setCurrentIndex(1)  # F1 (indirect)

    canvas = window.processing_preview_canvas
    assert canvas._contour_spectrum is None
    assert len(canvas.ax.lines) >= 1
    # Two panels now, not one: F1's interferogram above its spectrum,
    # so Truncate (F1) and Zero-fill (F1) can be judged against the
    # decay they are decisions about. The title names the FID panel.
    assert canvas._processing_preview_fid_ax is not None
    assert "F1 interferogram" in canvas._processing_preview_fid_ax.get_title()


def test_main_window_2d_indirect_processing_preview_does_not_double_apply_phase(qapp, tmp_path, monkeypatch):
    # A first version of _show_2d_indirect_processing_preview extracted
    # an F1 trace straight from spectrum.data -- already kept current
    # with the live phase delta by _apply_phase_preview's own "2d"
    # branch, see that method's docstring -- and then applied
    # phase0_f1/phase1_f1 plus the live spin-box delta *again* on top,
    # silently double-phasing the preview. Found by cross-checking
    # against a direct get_trace read of spectrum.data (the already-
    # trusted source -- self.canvas no longer renders a 2D phase slice
    # of its own at all, see test_main_window_2d_phase_stays_a_contour_not_a_slice,
    # so that's the only "trusted reference" left) on real data; this
    # reproduces that same cross-check on a small synthetic fixture so
    # it runs in the regular suite.
    #
    # Deliberately not the fixture's own default fid (identical across
    # every F1 row): that F1 FFTs to a pure single-point real delta,
    # and MplCanvas's own real-part display (NMRSpectrum.normalized(),
    # scaled to [-1, 1] by its own max-abs) reduces any single-nonzero-
    # point array to the same [0, 0, +-1, 0] shape regardless of that
    # one point's actual phase/magnitude -- rendered-line comparisons
    # against a fixture like that can't discriminate a phase bug at
    # all (confirmed directly: this test still passed with the double-
    # phasing bug reintroduced, using the default fid). A distinct
    # complex weight per row keeps the F1 trace multi-point and
    # genuinely phase-sensitive after normalization.
    n_f1, n_f2 = 4, 8
    t2 = np.arange(n_f2) / 8000.0
    base_row = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000.0 * t2)
    weights = np.array([1.0, 0.6 + 0.3j, -0.4 + 0.7j, 0.2 - 0.9j])
    fid = (weights[:, None] * base_row[None, :]).astype(np.complex128)

    window = MainWindow()
    window._load(
        _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=n_f1, n_f2=n_f2, f1_nuc="13C", fid=fid), vendor="bruker",
    )
    window.process_indirect_button.setChecked(True)
    window.dim_combo.setCurrentIndex(1)  # F1 (indirect)
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.phase0_handle.setValue(40.0)
    window.phase1_handle.setValue(15.0)

    reference_trace = get_trace(spectrum, axis="f2", index=window._phase_2d_slice_index)
    reference_y = reference_trace.normalized()

    canvas = window.processing_preview_canvas
    assert canvas._contour_spectrum is None
    preview_y = canvas.ax.lines[0].get_ydata()

    np.testing.assert_allclose(preview_y, reference_y, atol=1e-8)


def test_main_window_2d_pivot_matches_selected_dimension(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt out of the pseudo-2D default
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    f2_ppm = spectrum.axis_values(-1, "ppm")
    assert window.pivot_spin.value() == pytest.approx((f2_ppm.min() + f2_ppm.max()) / 2, abs=1e-3)

    window.dim_combo.setCurrentIndex(1)
    f1_ppm = spectrum.axis_values(0, "ppm")
    assert window.pivot_spin.value() == pytest.approx((f1_ppm.min() + f1_ppm.max()) / 2, abs=1e-3)


def test_main_window_pseudo2d_phase_affects_only_selected_slice(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.apply_all_slices_checkbox.setChecked(False)  # per-slice is what this test exercises
    index = window._selected_index()
    original = _complex(window.session.spectra[index]).copy()

    window.slice_spin.setValue(2)
    window.phase0_handle.setValue(45.0)

    updated = _complex(window.session.spectra[index])
    for row in range(original.shape[0]):
        if row == 1:  # slice 2, 1-based
            assert not np.allclose(original[row], updated[row])
            np.testing.assert_allclose(np.abs(updated[row]), np.abs(original[row]), atol=1e-6)
        else:
            np.testing.assert_allclose(updated[row], original[row], atol=1e-6)


def test_main_window_phase_from_contour_preserves_ppm_zoom(qapp, tmp_path):
    # Regression test: phasing a pseudo-2D spectrum from its default
    # contour view used to silently reset any ppm/index zoom the user
    # had set. self.canvas now always stays a contour and only ever
    # gets a debounced refresh during a drag (_schedule_canvas_2d_refresh
    # -- see its own docstring for why: a full contour rebuild has no
    # in-place update path, and resets the view by default), so both
    # axes' own zoom must be explicitly restored around that rebuild,
    # not just x -- unlike the old single-slice-line view (only x, ppm,
    # carried real meaning there; y was slice-index/intensity, a
    # different axis each time), a contour's y axis is a *second* ppm/
    # index axis just as meaningful to preserve as x.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    assert window.canvas._contour_spectrum is not None  # confirms we're really on the contour

    window.canvas.ax.set_xlim(2.0, 4.0)
    window.canvas.ax.set_ylim(1.0, 3.0)

    window.phase0_handle.setValue(10.0)
    _settle_canvas_2d_refresh(window)

    assert window.canvas._contour_spectrum is not None  # still a contour, not reset to something else
    assert window.canvas.ax.get_xlim() == pytest.approx((2.0, 4.0))
    assert window.canvas.ax.get_ylim() == pytest.approx((1.0, 3.0))


def test_main_window_pseudo2d_apply_to_all_slices(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    index = window._selected_index()
    original = _complex(window.session.spectra[index]).copy()

    window.apply_all_slices_checkbox.setChecked(True)
    window.phase0_handle.setValue(45.0)

    updated = _complex(window.session.spectra[index])
    assert not np.allclose(original, updated)
    np.testing.assert_allclose(np.abs(updated), np.abs(original), atol=1e-6)


def test_main_window_autophase_with_apply_all_phases_slices_independently(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    index = window._selected_index()

    # Distort every slice's phase first, so autophase has something
    # real to correct regardless of how well-phased the synthetic
    # fixture data already was.
    window.apply_all_slices_checkbox.setChecked(True)
    window.phase0_handle.setValue(65.0)
    distorted = _complex(window.session.spectra[index]).copy()

    # One Autophase button; "Apply to every slice" is what makes it
    # per-slice rather than this-slice-only.
    window._on_autophase_clicked()

    updated = _complex(window.session.spectra[index])
    assert not np.allclose(distorted, updated)
    # A pure phase correction never changes magnitude, per row.
    np.testing.assert_allclose(np.abs(updated), np.abs(distorted), atol=1e-6)
    # Autophase (maximizing each row's own real-part integral) should
    # improve on -- never worsen -- the distorted starting point.
    for row in range(updated.shape[0]):
        assert np.sum(np.real(updated[row])) >= np.sum(np.real(distorted[row])) - 1e-6


def test_main_window_apply_all_slices_checkbox_hidden_for_1d(qapp, tmp_path):
    # There is exactly one Autophase button, and the per-slice option
    # it offers is this checkbox -- which has nothing to mean on a 1D.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.apply_all_slices_checkbox.isHidden() is True
    assert not hasattr(window, "autophase_slices_button")


def test_main_window_pseudo2d_unchecking_apply_all_reverts_other_slices(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    index = window._selected_index()
    original = _complex(window.session.spectra[index]).copy()

    window.apply_all_slices_checkbox.setChecked(True)
    window.phase0_handle.setValue(45.0)
    window.apply_all_slices_checkbox.setChecked(False)

    updated = _complex(window.session.spectra[index])
    for row in range(original.shape[0]):
        if row == 0:  # slice_spin still at its default, slice 1
            assert not np.allclose(original[row], updated[row])
        else:
            np.testing.assert_allclose(updated[row], original[row], atol=1e-6)


def test_main_window_slice_prev_next_buttons_step_through_slices(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    assert window.slice_spin.value() == 1

    window.slice_next_button.click()
    assert window.slice_spin.value() == 2

    window.slice_next_button.click()
    window.slice_next_button.click()
    window.slice_next_button.click()
    window.slice_next_button.click()  # past the last slice (4) -- must clamp, not raise
    assert window.slice_spin.value() == 4

    window.slice_prev_button.click()
    assert window.slice_spin.value() == 3

    for _ in range(5):
        window.slice_prev_button.click()  # past slice 1 -- must clamp, not raise
    assert window.slice_spin.value() == 1


def test_main_window_slice_prev_next_buttons_hidden_for_conventional_2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt out of the pseudo-2D default
    assert window.slice_prev_button.isHidden() is True
    assert window.slice_next_button.isHidden() is True


def test_main_window_pseudo2d_switching_slice_resets_phase_controls(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")

    window.phase0_handle.setValue(45.0)
    window.slice_spin.setValue(2)

    assert window.phase0_spin.value() == pytest.approx(0.0)
    assert window.phase0_handle.value() == pytest.approx(0.0)


def test_main_window_pseudo2d_phase_sets_processing_preview_slice_title(qapp, tmp_path, monkeypatch):
    # Renamed from this test's own former name,
    # "..._sets_canvas_slice_title": self.canvas always stays a
    # stacked/contour view now (see
    # test_main_window_2d_phase_stays_a_contour_not_a_slice) -- the
    # titled slice this test checks for lives on the Processing-preview
    # canvas instead. _make_raw_2d_dataset -- see that same test's own
    # comment on why the plain fixture can't exercise the Processing-
    # preview canvas at all (_on_side_tab_changed's own
    # _fid_reprocessable gate).
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8), vendor="bruker")
    window.slice_spin.setValue(3)
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    window.phase0_handle.setValue(10.0)

    # The title lands on the dual-panel preview's own FID (top) axes,
    # not processing_preview_canvas.ax itself (== the bottom spectrum
    # axes, spec_ax -- see show_processing_preview's own layout).
    assert window.processing_preview_canvas._processing_preview_fid_ax.get_title() == "Slice 3"
    assert window.canvas._contour_spectrum is not None  # unaffected -- still not a slice


def test_slice_label_reports_bare_position_when_index_matches_it(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    assert MainWindow._slice_label(spectrum, 2) == "Slice 2"


def test_slice_label_shows_recorded_index_when_it_differs_from_position():
    axis = NMRAxis(kind="index", index=np.array([10, 20, 30]))
    freq_axis = NMRAxis(kind="frequency", ppm=np.linspace(0, 1, 4))
    spectrum = NMRSpectrum(
        data=np.zeros((3, 4)), axes=[axis, freq_axis],
        metadata=NMRMetadata(vendor="bruker", pseudo_2d=True),
    )
    assert MainWindow._slice_label(spectrum, 2) == "Slice 2 (index 20)"


# ============================================================
# FID processing (pre-FFT reprocessing + FID preview)
# ============================================================

def _synthetic_raw_fid(n=64, sw=8000.0, f0_hz=1000.0, decay=200.0):
    t = np.arange(n) / sw
    return (np.exp(-t * decay) * np.exp(2j * np.pi * f0_hz * t)).astype(np.complex128)


def _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="rawexpt1", probe=None, pulprog=None):
    """A Bruker experiment with a real (parseable) `procs` but a mocked
    raw `fid` read -- same approach as test_bruker_raw_fid.py, so
    `read_1d`'s real window/zero-fill/FFT code runs for real while the
    two genuinely nmrglue-owned calls (`ng.bruker.read`,
    `remove_digital_filter`) are stubbed with controlled data."""
    from opennmr.io import bruker as bruker_io

    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n, "SF": 400.13, "WDW": 0, "LB": 0.0, "SSB": 0.0})
    (expt / "fid").write_bytes(b"\x00")  # existence is all _read_auto checks; content is never read (mocked)
    (expt / "acqus").write_bytes(b"\x00")

    acqus = {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1.0}
    if probe is not None:
        acqus["PROBHD"] = probe
    if pulprog is not None:
        acqus["PULPROG"] = pulprog
    fid = _synthetic_raw_fid(n=n)
    dic = {"acqus": acqus}
    # read_acqus_file too: read_1d/read_2d take their parameters from it
    # now, rather than from a combined ng.bruker.read call. Without this
    # the load raises on the placeholder acqus these fixtures write, and
    # _load answers with a modal QMessageBox -- which hangs the suite
    # rather than failing it.
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    monkeypatch.setattr(bruker_io, "_read_bruker_binary", lambda p, d, shape=None: fid)
    monkeypatch.setattr(bruker_io, "_remove_digital_filter", lambda d, data: data)
    return str(expt)


def _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=32, name="rawexpt2d", f1_nuc="13C", ns=1.0,
                          fn_mode=None, fid=None, f1_o1=0.0):
    from opennmr.io import bruker as bruker_io

    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13, "WDW": 0})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 100.6, "WDW": 0})
    (expt / "ser").write_bytes(b"\x00")
    (expt / "acqus").write_bytes(b"\x00")

    acqus = {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": ns}
    # `acqu2s` is read from the mocked `ng.bruker.read()` dict, not
    # parsed off disk -- matching real Bruker behaviour (nmrglue reads
    # every acqu*s/proc*s file itself as part of one `read()` call).
    acqu2s = {"NUC1": f1_nuc, "BF1": 100.6, "O1": f1_o1}
    if fn_mode is not None:
        acqu2s["FnMODE"] = fn_mode
    if fid is None:
        t2 = np.arange(n_f2) / 8000.0
        row = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000.0 * t2)
        fid = np.tile(row, (n_f1, 1)).astype(np.complex128)
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    # read_acqus_file too: read_1d/read_2d take their parameters from it
    # now, rather than from a combined ng.bruker.read call. Without this
    # the load raises on the placeholder acqus these fixtures write, and
    # _load answers with a modal QMessageBox -- which hangs the suite
    # rather than failing it.
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    monkeypatch.setattr(bruker_io, "_read_bruker_binary", lambda p, d, shape=None: fid)
    monkeypatch.setattr(bruker_io, "_remove_digital_filter", lambda d, data: data)
    # _guess_raw_2d_shape's real-file-scale TD/DTYPA/file-size logic
    # (see its own docstring in io/bruker.py) doesn't apply to this
    # small synthetic fixture -- bypass it so the mocked `fid` above
    # passes straight through `_read_raw_2d_data` at its own actual
    # shape, unchanged.
    monkeypatch.setattr(bruker_io, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)
    return str(expt)


# ============================================================
# Proc number selection (still reads raw ser/fid by default)
# ============================================================

def test_main_window_change_procno_reloads_with_new_procno(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="rawexpt_procno")
    # A second proc number with a different recorded window --
    # observable proof the reload actually picked it up, while the
    # raw fid itself (mocked, shared across procnos) is untouched.
    pdata2 = tmp_path / "rawexpt_procno" / "pdata" / "2"
    pdata2.mkdir(parents=True)
    _write_params(pdata2 / "procs", {"SW_p": 8000.0, "SI": 64, "SF": 400.13, "WDW": 1, "LB": 50.0, "SSB": 0.0})

    window._load(path, vendor="bruker")
    index = window._selected_index()
    assert window.session.spectra[index].metadata.procno == 1
    original = window.session.spectra[index].data.copy()

    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (2, True))
    window._on_change_procno_action(index)

    updated = window.session.spectra[index]
    assert updated.metadata.procno == 2
    assert not np.allclose(original, updated.data)
    assert "procno 2" in window.spectrum_list.item(index).text()


def test_main_window_change_procno_cancelled_is_noop(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    index = window._selected_index()
    original = window.session.spectra[index].data.copy()

    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (2, False))
    window._on_change_procno_action(index)

    assert window.session.spectra[index].metadata.procno == 1
    np.testing.assert_allclose(window.session.spectra[index].data, original)


class _FakeAction(str):
    """Stands in for a QAction where a fake menu is used.

    Compares equal to its own label, so every ``"Some label" in
    added_labels`` assertion keeps working, while tolerating the
    QAction API the real menu code calls on what ``addAction``
    returns."""

    def setToolTip(self, _text) -> None:
        pass

    def setCheckable(self, _value) -> None:
        pass

    def setChecked(self, _value) -> None:
        pass


def test_main_window_change_procno_not_offered_for_xpulse(qapp, tmp_path, monkeypatch):
    # QMenu.exec() is modal and not interceptable headlessly by
    # monkeypatching the real PySide6 class (see
    # test_main_window_spectrum_list_context_menu_wires_actions's own
    # comment) -- replace the name main_window.py actually calls
    # (`QMenu(self)`) with a lightweight fake instead.
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    index = window._selected_index()

    added_labels = []

    class _FakeMenu:
        def __init__(self, *a, **k):
            pass

        def addAction(self, label):
            added_labels.append(label)
            return _FakeAction(label)

        def addSeparator(self):
            pass

        def exec(self, *a, **k):
            return None

    import opennmr.gui.main_window as main_window_module
    monkeypatch.setattr(main_window_module, "QMenu", _FakeMenu)

    pos = window.spectrum_list.visualItemRect(window.spectrum_list.item(index)).center()
    window._on_spectrum_list_context_menu(pos)

    assert "Change proc number…" not in added_labels


def test_main_window_fid_processing_box_hidden_for_processed_only_data(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.fid_processing_box.isHidden() is True


def test_main_window_fid_processing_box_visible_for_raw_fid_data(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    assert window.fid_processing_box.isHidden() is False


def test_main_window_window_change_updates_spectrum_live(qapp, tmp_path, monkeypatch):
    # No more "Apply Processing" button -- every FID-processing
    # control writes straight into spectrum.data live (explicit,
    # repeated user feedback: a separate commit step kept resetting
    # unrelated state).
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()
    original = window.session.spectra[index].data.copy()

    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)

    updated = window.session.spectra[index].data
    assert len(updated) == 64  # unchanged -- zero-fill wasn't touched, defaults to the as-loaded length
    assert not np.allclose(original, updated)
    assert window.session.spectra[index].metadata.raw.get("fid") is not None


def test_main_window_zero_fill_change_updates_spectrum_length_live(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()
    assert len(window.session.spectra[index].data) == 64

    window._set_zero_fill_value(128)  # sets the widgets only, no signal (see its own docstring)
    window._on_fid_processing_control_changed()  # the change this would trigger from a real UI interaction

    assert len(window.session.spectra[index].data) == 128


def test_main_window_zoom_survives_a_window_function_change(qapp, tmp_path, monkeypatch):
    # Reported directly: "there are still instances where the view is
    # being reset, probably when the fid is being recomputed after
    # windowing etc., can the stored view just be applied to the new
    # plot each time?"
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    canvas = window.canvas
    canvas.ax.set_xlim(6.0, 4.0)
    canvas.ax.set_ylim(0.1, 0.4)

    hann = next(i for i in range(window.window_combo.count())
                if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann)

    assert canvas.ax.get_xlim() == pytest.approx((6.0, 4.0))
    assert canvas.ax.get_ylim() == pytest.approx((0.1, 0.4))


def test_main_window_zoom_survives_a_zero_fill_change(qapp, tmp_path, monkeypatch):
    # The harder half: zero-fill changes the point count, which sends
    # update_data down its rebuild path. That path restored x only, so
    # the same edit kept or discarded the intensity zoom depending
    # purely on whether the length happened to change.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    canvas = window.canvas
    canvas.ax.set_xlim(6.0, 4.0)
    canvas.ax.set_ylim(0.1, 0.4)

    window._set_zero_fill_value(128)
    window._on_fid_processing_control_changed()

    assert len(window.session.spectra[0].data) == 128
    assert canvas.ax.get_xlim() == pytest.approx((6.0, 4.0))
    assert canvas.ax.get_ylim() == pytest.approx((0.1, 0.4))


def test_main_window_processing_preview_zoom_survives_a_control_change(qapp, tmp_path, monkeypatch):
    # Same complaint, on the tab where FID processing is actually done.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    preview = window.processing_preview_canvas
    assert preview._processing_preview_spec_ax is not None

    preview._processing_preview_spec_ax.set_xlim(6.0, 4.0)

    def flat():
        return [v for panel in preview.processing_preview_limits() for lim in panel for v in lim]

    before = flat()

    hann = next(i for i in range(window.window_combo.count())
                if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann)

    assert flat() == pytest.approx(before)


def test_main_window_clicking_fid_tab_switches_canvas_to_time_domain(qapp, tmp_path, monkeypatch):
    # No more "Preview as FID" checkbox -- clicking the FID tab itself
    # is what shows it (and switching back to Spectrum shows that).
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")

    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    assert window.canvas_tabs.currentWidget() is window.fid_canvas
    assert window.fid_canvas.ax.get_xlabel() == "Time (s)"
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Real" in labels  # default preview mode

    window.canvas_tabs.setCurrentWidget(window.canvas)
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_fid_tab_clears_without_raw_fid(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    assert len(window.fid_canvas.ax.lines) == 0


def test_main_window_fid_tab_clears_for_conventional_2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    # acqu2s's NUC1="13C" (not "off"), so auto-detection would treat
    # this as a conventional heteronuclear 2D -- but pseudo-2D is now
    # the default regardless, so opt back out explicitly to get a
    # genuine conventional-2D spectrum for this test.
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    assert not window.session.spectra[window._selected_index()].is_pseudo_2d
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    assert len(window.fid_canvas.ax.lines) == 0


def test_main_window_fid_tab_shows_preview_for_pseudo_2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Real" in labels  # default preview mode


def test_main_window_fid_preview_toggle_shows_magnitude_only(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    window.fid_display_combo.setCurrentIndex(1)  # Magnitude
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Magnitude" in labels
    assert "Real" not in labels


def test_main_window_fid_preview_tracks_selected_slice(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)

    window.slice_spin.setValue(3)
    assert window.fid_canvas.ax.get_title() == "Slice 3"


def test_main_window_fid_x_unit_combo_switches_axis_label(qapp, tmp_path, monkeypatch):
    # Explicit user feedback: FID intensity should be plotted against
    # time (with units), with points as a secondary option -- the
    # underlying x data was already time-based, but nothing ever
    # labelled the axis at all (see show_processing_preview's own fix);
    # this combo is the user-facing switch between the two.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)

    assert window.fid_x_unit_combo.currentData() == "time"
    assert window.fid_canvas.ax.get_xlabel() == "Time (s)"

    points_row = next(
        i for i in range(window.fid_x_unit_combo.count()) if window.fid_x_unit_combo.itemData(i) == "points"
    )
    window.fid_x_unit_combo.setCurrentIndex(points_row)
    assert window.fid_canvas.ax.get_xlabel() == "Points"


def test_main_window_phase_current_display_wide_enough_for_large_values(qapp):
    # Explicit user feedback: the Phase 0/1 controls were too narrow
    # to read a value like "-1234.56" -- still relevant now that these
    # are read-only current-value displays rather than editable spin
    # boxes (see _make_phase_control's own comment), since a real
    # recorded phase1 can run into 4 digits.
    window = MainWindow()
    for display in (window.phase0_current_display, window.phase1_current_display):
        assert display.minimumWidth() >= 80


# ============================================================
# Processing tab's own live dual preview (FID + downsampled spectrum)
# ============================================================

def test_main_window_apply_processing_button_removed_in_favour_of_reset(qapp):
    # Explicit, repeated user feedback: "Apply Processing is still
    # resetting everything... remove the button, we can have a reset
    # button" -- every FID-processing control is live now (see
    # _on_fid_processing_control_changed), so there's no "apply" step
    # left to have a button for; Reset (the opposite -- discard, not
    # commit) replaces it.
    window = MainWindow()
    buttons = window.fid_processing_box.findChildren(QPushButton)
    assert not any(b.text() == "Apply Processing" for b in buttons)
    assert not any("Reprocess" in b.text() for b in buttons)
    assert any(b.text() == "Reset" for b in buttons)


def test_main_window_selecting_processing_tab_shows_processing_preview(qapp, tmp_path, monkeypatch):
    # Explicit user request: while on the Processing tab, the canvas
    # should switch over to its own dual FID+spectrum preview
    # automatically, unprompted.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")

    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas
    fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    assert len(fid_ax.lines) >= 1
    assert len(_trace_lines(spec_ax)) == 1


def test_main_window_leaving_processing_tab_switches_back_to_spectrum(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas

    window.side_tabs.setCurrentIndex(0)  # back to Spectra
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_leaving_processing_tab_respects_manual_canvas_choice(qapp, tmp_path, monkeypatch):
    # If the user manually navigated elsewhere (e.g. FID) while still
    # on the Processing tab, leaving it shouldn't yank them away from
    # that choice -- only switch back if the preview was actually the
    # one showing.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)

    window.side_tabs.setCurrentIndex(0)
    assert window.canvas_tabs.currentWidget() is window.fid_canvas


def test_main_window_processing_preview_hidden_for_non_reprocessable_spectrum(qapp, tmp_path):
    # A plain (already-processed, no raw fid/ser) Bruker spectrum has
    # nothing for the FID-processing box to preview -- same gate as
    # that box's own visibility.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window.canvas_tabs.indexOf(window.processing_preview_canvas)
    assert window.canvas_tabs.isTabVisible(index) is False


def test_main_window_processing_preview_tracks_selected_slice(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas

    window.slice_spin.setValue(3)
    fid_ax, _spec_ax = window.processing_preview_canvas.figure.axes
    assert fid_ax.get_title() == "Slice 3"


def test_main_window_processing_preview_and_spectrum_update_together_live(qapp, tmp_path, monkeypatch):
    # There's no "Apply Processing" step any more -- tweaking a FID-
    # processing control updates the Processing-preview tab AND the
    # actually-loaded spectrum (so the Spectrum tab too) together, from
    # the same source, live. Explicit, repeated user feedback: a
    # separate not-yet-applied preview vs. a stale "actual" spectrum
    # was exactly the split that kept losing state on commit.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()
    spectrum_data_before = window.session.spectra[index].data.copy()

    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    preview_before = spec_ax.lines[0].get_ydata().copy()

    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)

    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    preview_after = spec_ax.lines[0].get_ydata()
    assert not np.allclose(preview_before, preview_after)

    spectrum_after = window.session.spectra[index]
    assert not np.allclose(spectrum_data_before, spectrum_after.data)
    # Same source: the preview's own (downsampled, normalized-for-
    # display) data and the actual spectrum's own normalized() agree
    # at every point the preview kept (no downsampling here -- 64
    # points is under plot_1d's own downsampling threshold).
    np.testing.assert_allclose(spectrum_after.normalized(), preview_after, atol=1e-6)


def test_main_window_processing_preview_never_reads_from_disk(qapp, tmp_path, monkeypatch):
    # The actual performance fix: the preview used to call
    # Session.preview() (io.load() under the hood -- real file I/O,
    # reprocessing every slice of a pseudo-2D just to show one) on
    # every live control change. It's now computed entirely from the
    # already-cached raw FID (metadata.raw["fid_raw"]) -- assert
    # io.load is never even called for a live preview refresh, not
    # just that the result looks plausible.
    import opennmr.io as io_module

    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")

    # Recorded (not raised) -- _show_processing_preview's own
    # best-effort try/except would otherwise silently swallow a raised
    # assertion, defeating the point of this check.
    calls = []

    def _record_call(*a, **k):
        calls.append((a, k))
        raise RuntimeError("stub -- a real call would still be a bug even though this stops it")

    monkeypatch.setattr(io_module, "load", _record_call)

    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.slice_spin.setValue(3)
    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)
    window.phase0_handle.setValue(30.0)

    assert calls == []


def test_main_window_processing_preview_pseudo2d_uses_only_active_slice(qapp, tmp_path, monkeypatch):
    # _make_raw_2d_dataset tiles the same row for every slice (fine
    # for tests that don't care about per-slice content) -- built with
    # genuinely varying rows here instead, so the assertion below
    # actually exercises "extracts the selected slice's own data," not
    # just "doesn't crash."
    from opennmr.io import bruker as bruker_io

    expt = tmp_path / "rawexpt2d_varying"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    n_f1, n_f2 = 4, 32
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13, "WDW": 0})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 100.6, "WDW": 0})
    (expt / "ser").write_bytes(b"\x00")
    (expt / "acqus").write_bytes(b"\x00")
    acqus = {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "off", "BF1": 100.6, "O1": 0.0}
    t2 = np.arange(n_f2) / 8000.0
    # A different decay rate per row -- slices are visibly distinct.
    fid = np.stack([np.exp(-t2 * 300 * (row + 1)) * np.exp(2j * np.pi * 1000.0 * t2) for row in range(n_f1)])
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    # read_acqus_file too: read_1d/read_2d take their parameters from it
    # now, rather than from a combined ng.bruker.read call. Without this
    # the load raises on the placeholder acqus these fixtures write, and
    # _load answers with a modal QMessageBox -- which hangs the suite
    # rather than failing it.
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    monkeypatch.setattr(bruker_io, "_read_bruker_binary", lambda p, d, shape=None: fid)
    monkeypatch.setattr(bruker_io, "_remove_digital_filter", lambda d, data: data)
    monkeypatch.setattr(bruker_io, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)

    window = MainWindow()
    window._load(str(expt), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    window.slice_spin.setValue(1)
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    slice1 = spec_ax.lines[0].get_ydata().copy()

    window.slice_spin.setValue(3)
    _fid_ax, spec_ax = window.processing_preview_canvas.figure.axes
    slice3 = spec_ax.lines[0].get_ydata()

    assert not np.allclose(slice1, slice3)  # genuinely tracks the selected slice, not a cached/stale one


# ============================================================
# TDeff (X-Pulse raw-FID truncation) control
# ============================================================

_XPULSE_FIXTURE = """\
##TITLE= synthetic test fixture
##$SFO1= 400.130000
##$SF= 400.130000
##$O1= 2400.500000
##.OBSERVENUCLEUS= 1H
##VARFORM= AFFN, AFFN, AFFN
##LAST= 0.000300
$$ SPINFLOW START
$$NP=4
$$Filter=10000
$$TDEFF=3
$$SI=8
$$ SPINFLOW END
##PAGE= N=1
##DATATABLE= (X++(R..R))
0.000000 1.0
0.000100 0.9
0.000200 0.8
0.000300 0.7
##PAGE= N=2
##DATATABLE= (X++(R..R))
0.000000 0.0
0.000100 0.0
0.000200 0.0
0.000300 0.0
##END=
"""


def _make_xpulse_dataset(tmp_path, name="xpulse1.jdx"):
    path = tmp_path / name
    path.write_text(_XPULSE_FIXTURE)
    return str(path)


def _make_xpulse_array_experiment(tmp_path, n_steps=3, name="pfg_experiment"):
    """A parent folder holding ``array1``..``arrayN`` subfolders, each
    with the same synthetic 1D fixture -- mirrors X-Pulse's real PFG/
    T1/T2 array layout (see ``opennmr.io.xpulse.read_array``), so
    ``metadata.raw["fid_raw"]`` ends up genuinely 2D: ``(n_steps,
    NP)``, not the plain 1D shape a single-file X-Pulse load produces."""
    root = tmp_path / name
    for i in range(1, n_steps + 1):
        array_dir = root / f"array{i}"
        array_dir.mkdir(parents=True)
        (array_dir / f"sample_{i}_avg.jdx").write_text(_XPULSE_FIXTURE)
    return str(root)


def _make_xpulse_ntuples_2d_dataset(tmp_path, n1=4, n2=32, name="hsqc.jdx"):
    """A minimal synthetic NTUPLES 2D fixture -- deliberately
    asymmetric (n1 << n2, matching the real HSQC test dataset's own
    128x2048 shape) since that's exactly the shape that exposed the
    real fft_spectrum bug (len() reading axis 0 instead of the last
    axis for a 2D array) as a loud crash rather than a silent
    wrong-value bug a square array wouldn't have caught."""
    t2 = np.arange(n2) / 800.0
    # TPPI, and R/I are the real and imaginary halves of ONE complex t2
    # FID -- see tests/test_xpulse.py::_make_ntuples_fixture for the
    # full reasoning. The old model here built States cos/sin pairs of
    # a real signal, which agreed with what the reader then did rather
    # than with what the files contain.
    decay = np.exp(-t2 * 5.0)
    # n1 counts acquired FIDs; F1's States quadrature uses two per t1
    # increment, so there are n1 // 2 increments and 2 * n1 pages.
    n_t1 = max(n1 // 2, 1)
    f1_hz, f2_hz = 100.0 / n_t1, 2 * (800.0 / n2)
    lines = [
        "##TITLE= synthetic NTUPLES 2D fixture", "##DATACLASS= NTUPLES", "##NUMDIM= 2",
        "$$SpinFlow Sequence Parameters Start",
        f"$$NP1= {n1}", f"$$NP2= {n2}", "$$SW1= 100.0", "$$Filter= 800.0",
        "$$O1= 0.0", "$$NS= 1", "$$TxPPM= 5.0", "$$TxPPMX= 100.0",
        "$$SpinFlow Sequence Parameters End",
        "##.OBSERVENUCLEUS= ^1H", "##$INDIRECTNUCLEUS= ^13C",
        "##.ACQUISITION SCHEME= TPPI",
        "##$SF= 500.0 $$ f2", "##$SF= 125.0 $$ f1",
        "##$SFO1= 500.0 $$ f2", "##$SFO1= 125.0 $$ f1",
        "##NTUPLES= nD NMR FID", "##VARFORM= AFFN, AFFN, AFFN, AFFN",
        "##FIRST= 0, 0, 0, 0", f"##LAST= {(n1 - 1) / 100.0}, {(n2 - 1) / 800.0}, 0, 0",
        "##FIRST= 0,0,0,0",
    ]
    for k in range(n_t1):
        sign = -1.0 if k % 2 else 1.0          # States-TPPI alternation
        for quadrature, sense in ((np.cos, 1.0), (np.sin, -1.0)):
            amplitude = sign * sense * quadrature(2 * np.pi * f1_hz * (k / 100.0))
            complex_row = amplitude * decay * np.exp(-2j * np.pi * f2_hz * t2)
            for kind, row in (("R", np.real(complex_row)), ("I", np.imag(complex_row))):
                lines.append(f"##PAGE=\tT1= {k}")
                lines.append("##FIRST= 0,0,0,0")
                lines.append(f"##DATATABLE= (T2++({kind}..{kind})), PROFILE")
                for t2_val, value in zip(t2, row):
                    lines.append(f"{t2_val:.6f} {value:.6f}")
    lines.append("##END=")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_main_window_xpulse_2d_loads_and_renders_without_crashing(qapp, tmp_path, monkeypatch):
    # Real bug, found on real data (a 128x2048 HSQC): loading a
    # genuine X-Pulse 2D (NTUPLES) spectrum crashed while rendering
    # the initial contour -- ValueError: x and y must have same first
    # dimension -- caught by _draw_selected_spectrum's own try/except
    # and turned into a QMessageBox.critical, which blocks forever in
    # a headless test with nothing to click "OK" (a real, separate
    # trap this test avoids by mocking it, same as the load itself
    # would show a real user). Root cause: processing.ft.fft_spectrum
    # used len(fid) (reads axis 0) instead of fid.shape[-1] for a 2D
    # array's own frequency-axis length -- fixed at the source.
    from PySide6.QtWidgets import QMessageBox

    path = _make_xpulse_ntuples_2d_dataset(tmp_path)
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical_calls.append(a))

    window = MainWindow()
    window._load(path, vendor="xpulse")

    assert critical_calls == []
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    # 4 acquired FIDs are 2 States (cos, sin) t1 increments.
    assert spectrum.data.shape[0] == 2
    assert len(spectrum.axes[0].ppm) == spectrum.data.shape[0]
    assert len(spectrum.axes[1].ppm) == spectrum.data.shape[1]


def test_main_window_xpulse_2d_phase_drag_does_not_crash(qapp, tmp_path, monkeypatch):
    # The user's own real report: the crash above happened repeatedly
    # during a live phase drag specifically (_apply_phase_preview's
    # "2d" branch), not just once on load -- that call site had no
    # exception guard at all (unlike _draw_selected_spectrum's), so it
    # printed an unhandled traceback once per mouse-move tick for the
    # whole drag. Both the underlying shape bug and the missing guard
    # are fixed; this exercises the live-drag path specifically.
    from PySide6.QtWidgets import QMessageBox

    path = _make_xpulse_ntuples_2d_dataset(tmp_path)
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical_calls.append(a))

    window = MainWindow()
    window._load(path, vendor="xpulse")
    assert window._phase_mode == "2d"

    window.phase0_handle.setValue(15.0)

    assert critical_calls == []
    index = window._selected_index()
    assert window.session.spectra[index].data.shape[0] == 2  # 4 FIDs -> 2 t1 points


def test_main_window_process_indirect_button_works_for_xpulse_2d(qapp, tmp_path, monkeypatch):
    # Real user report: "Importing 2D x-pulse causes a crash." Root
    # cause: io.xpulse.read_2d had no pseudo_2d parameter, so reaching
    # _on_process_indirect_toggled's reload raised "TypeError:
    # read_2d() got an unexpected keyword argument 'pseudo_2d'". That
    # was first fixed by *hiding* the button for this vendor -- which
    # stopped the crash but left a second report behind it: "some
    # pseudo-2Ds are being noted as regular 2Ds", with no way to
    # correct one. read_2d takes pseudo_2d now, so the toggle is shown
    # and works like every other vendor's.
    from PySide6.QtWidgets import QMessageBox

    path = _make_xpulse_ntuples_2d_dataset(tmp_path)
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical_calls.append(a))

    window = MainWindow()
    window._load(path, vendor="xpulse")

    assert critical_calls == []
    # Shown, and it round-trips without the TypeError that started all
    # this -- which is what the hiding was standing in for.
    assert not window.process_indirect_button.isHidden()
    # X-Pulse still *loads* a NTUPLES 2D as a genuine 2D (unlike
    # Bruker, whose auto-detection is overridden): every one in
    # evidence is a real correlation experiment. The toggle is the way
    # to correct a series that is not.
    assert not window.session.spectra[0].is_pseudo_2d
    window.process_indirect_button.setChecked(False)   # -> pseudo-2D
    assert critical_calls == []
    assert window.session.spectra[0].is_pseudo_2d
    window.process_indirect_button.setChecked(True)    # -> back to genuine 2D
    assert critical_calls == []
    assert not window.session.spectra[0].is_pseudo_2d

    # A genuine (non-pseudo) Bruker 2D must still show and support it.
    bruker_path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="stillworks2d", f1_nuc="13C")
    window._load(bruker_path, vendor="bruker")
    assert not window.process_indirect_button.isHidden()
    window.process_indirect_button.setChecked(True)
    assert critical_calls == []


def test_main_window_linear_predict_direction_defaults_to_forward_and_threads_through(qapp, tmp_path):
    # LP direction should actually change the live-recomputed spectrum
    # (not just be recorded) -- forward extends past the FID's own
    # measured end, backward reconstructs points before its start, so
    # the two must diverge once linear-predict points > 0.
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    assert window.linear_predict_direction_combo.currentData() == "forward"
    index = window._selected_index()

    window.linear_predict_points_spin.setValue(4)
    forward_data = window.session.spectra[index].data.copy()

    backward_index = window.linear_predict_direction_combo.findData("backward")
    window.linear_predict_direction_combo.setCurrentIndex(backward_index)
    backward_data = window.session.spectra[index].data.copy()

    assert not np.allclose(forward_data, backward_data)


def test_main_window_tdeff_spin_enabled_and_prefilled_for_xpulse(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    assert window.tdeff_spin.isEnabled() is True
    assert window.tdeff_spin.value() == 3  # prefilled from the file's own recorded TDEFF


def test_main_window_tdeff_spin_enabled_for_every_reprocessable_vendor(qapp, tmp_path, monkeypatch):
    # Reported directly: "'Truncate to:' in the pseudo-2D is generally
    # greyed out". It was gated to X-Pulse, even though
    # processing.ft.prepare_fid always supported truncation -- the
    # Bruker readers simply never exposed the parameter.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    assert window.tdeff_spin.isEnabled() is True
    # And its maximum is a real point count, not an arbitrary ceiling
    # the user would have to count down from.
    assert 1 < window.tdeff_spin.maximum() < (1 << 20)


@pytest.mark.parametrize("dataset", ["bruker_T2/11", "bruker_PFG/150"])
def test_main_window_tdeff_spin_enabled_for_a_real_bruker_pseudo_2d(qapp, dataset):
    # The exact case reported. Real data, because a raw ``ser`` on disk
    # is what makes a spectrum reprocessable at all -- a synthetic
    # in-memory 2D is correctly *not*, which is why this cannot be
    # checked against the usual fixture.
    path = os.path.join("test_data", *dataset.split("/"))
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    window = MainWindow()
    window._load(path, vendor="bruker")
    spectrum = window.session.spectra[0]
    assert spectrum.ndim == 2 and spectrum.is_pseudo_2d
    assert window.tdeff_spin.isEnabled() is True
    # Prefilled with the real per-FID point count, so stepping down
    # trims from a meaningful place.
    assert window.tdeff_spin.value() == window.tdeff_spin.maximum()
    assert window.tdeff_spin.maximum() == spectrum.metadata.raw["fid_raw"].shape[-1]


def test_main_window_tdeff_spin_disabled_when_there_is_no_raw_fid(qapp, tmp_path):
    # A processed-only load has nothing to truncate -- the control is
    # correctly unavailable there, which is the case the old X-Pulse
    # gate was standing in for.
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    assert window.tdeff_spin.isEnabled() is False


def test_main_window_tdeff_spin_defaults_to_full_raw_fid_length_when_no_tdeff_recorded(qapp, tmp_path):
    # Explicit user feedback: counting *up* from a small number to reach
    # a large, meaningful truncation point is painful -- the spin box
    # should default to (and let the user step down from) the full raw
    # FID length instead.
    path = tmp_path / "no_tdeff.jdx"
    path.write_text(_XPULSE_FIXTURE.replace("$$TDEFF=3\n", ""))
    window = MainWindow()
    window._load(str(path), vendor="xpulse")
    index = window._selected_index()
    raw_length = len(window.session.spectra[index].metadata.raw["fid_raw"])
    assert window.tdeff_spin.maximum() == raw_length
    assert window.tdeff_spin.value() == raw_length


def test_main_window_tdeff_spin_maximum_uses_per_fid_length_not_array_step_count(qapp, tmp_path):
    # Real user report, confirmed by direct reproduction: for an
    # arrayed X-Pulse dataset (a PFG/T1/T2 series), metadata.raw
    # ["fid_raw"] is genuinely 2D -- (n_steps, NP). The old
    # `len(fid_raw)` read axis 0 (the *step* count), so a real 16-step
    # series capped Truncate's own maximum at 16 regardless of how
    # many points were actually acquired per FID -- "I can't process
    # more than 16 data points." Deliberately using more array steps
    # (16) than the fixture's own per-FID point count (4) here, so the
    # old buggy behaviour (max == 16) and the fixed one (max == 4)
    # can't be confused with each other by coincidence.
    window = MainWindow()
    path = _make_xpulse_array_experiment(tmp_path, n_steps=16)
    window._load(path, vendor="xpulse")
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    assert spectrum.metadata.raw["fid_raw"].shape == (16, 4)
    assert window.tdeff_spin.maximum() == 4
    assert window.tdeff_spin.isEnabled() is True


def _make_varian_dataset(tmp_path, n=8, name="varian1.fid"):
    # Minimal round-trip via nmrglue's own writer, same approach as
    # tests/test_varian.py's _write_varian_1d -- avoids hand-fabricating
    # Varian's binary block-header format.
    import nmrglue as ng

    def entry(field_name, values, basictype="1"):
        return {
            "name": field_name, "subtype": "1", "basictype": basictype, "maxvalue": "0", "minvalue": "0",
            "stepsize": "0", "Ggroup": "0", "Dgroup": "0", "protection": "0", "active": "1",
            "intptr": "0", "values": [str(v) for v in values], "enumerable": "0", "enumerables": [],
        }

    data = (np.arange(1, n + 1) + 1j * np.arange(1, n + 1)).astype(np.complex64)
    dic = {
        "nblocks": 1, "ntraces": 1, "np": 2 * n, "ebytes": 4, "tbytes": 4 * 2 * n, "bbytes": 4 * 2 * n + 28,
        "vers_id": 0, "status": 201, "nbheaders": 1,
        "S_DATA": 1, "S_SPEC": 0, "S_32": 0, "S_FLOAT": 1, "S_COMPLEX": 0, "S_HYPERCOMPLEX": 0,
        "S_ACQPAR": 1, "S_SECND": 0, "S_TRANSF": 0, "S_NP": 0, "S_NF": 0, "S_NI": 0, "S_NI2": 0,
        "procpar": {
            "sw": entry("sw", [1000.0]), "sfrq": entry("sfrq", [100.0]), "reffrq": entry("reffrq", [100.0]),
            "tof": entry("tof", [0.0]), "tn": entry("tn", ["H1"], basictype="2"), "nt": entry("nt", [1.0]),
            "np": entry("np", [2 * n]), "ni": entry("ni", [1]),
        },
    }
    path = str(tmp_path / name)
    ng.varian.write(path, dic, data, overwrite=True)
    return path


def test_main_window_fid_processing_box_visible_for_varian(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_varian_dataset(tmp_path), vendor="varian")
    assert window.fid_processing_box.isHidden() is False


def test_main_window_unsupported_window_shows_notice_but_stays_reprocessable(qapp, tmp_path, monkeypatch):
    # Regression test: a Bruker WDW code this package doesn't (yet)
    # implement used to make the file unopenable entirely; now the FID
    # is loaded unwindowed and the user is told once, but the FID
    # Processing box must still be fully usable (not the earlier,
    # over-broad fix's behaviour of falling back to non-reprocessable
    # TopSpin-processed output and hiding the box).
    from opennmr.io import bruker as bruker_io

    expt = tmp_path / "gm_expt"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": 64, "SF": 400.13, "WDW": 2})
    (expt / "fid").write_bytes(b"\x00")
    (expt / "acqus").write_bytes(b"\x00")
    acqus = {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_raw_fid(n=64)
    dic = {"acqus": acqus}
    # read_acqus_file too: read_1d/read_2d take their parameters from it
    # now, rather than from a combined ng.bruker.read call. Without this
    # the load raises on the placeholder acqus these fixtures write, and
    # _load answers with a modal QMessageBox -- which hangs the suite
    # rather than failing it.
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    monkeypatch.setattr(bruker_io, "_read_bruker_binary", lambda p, d, shape=None: fid)
    monkeypatch.setattr(bruker_io, "_remove_digital_filter", lambda d, data: data)

    shown = []
    monkeypatch.setattr(
        "opennmr.gui.main_window.QMessageBox.information",
        lambda *a, **kw: shown.append(a[2] if len(a) > 2 else kw.get("text", "")),
    )

    window = MainWindow()
    window._load(str(expt), vendor="bruker")

    assert len(shown) == 1
    assert "WDW=2" in shown[0]
    index = window._selected_index()
    assert "window_unsupported" in window.session.spectra[index].metadata.raw
    assert window.fid_processing_box.isHidden() is False


def test_main_window_2d_shape_guess_failure_falls_back_and_hides_fid_box(qapp, tmp_path, monkeypatch):
    # Regression test, found on real 2D data (see TODO.md): nmrglue's
    # own shape-guessing can fail entirely for some real Bruker 2D
    # datasets (a genuine nmrglue bug, not something read_2d can work
    # around) -- io.bruker._read_auto now falls back to TopSpin's own
    # already-processed data instead of leaving the load unopenable,
    # and this box must correctly go from "usable" to "hidden" (unlike
    # the unsupported-window case above, there's no fid_raw cached at
    # all to reprocess from once this fallback happens).
    from opennmr.io import bruker as bruker_io

    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    (tmp_path / "expt2d" / "ser").write_bytes(b"\x00")

    def fake_read(*_a, **_kw):
        raise IndexError("tuple index out of range")

    # Patched at the binary-read seam, not ng.bruker.read: read_2d reads
    # the ser itself now (bruker._read_bruker_binary), so that is where a
    # raw-read failure has to be simulated for _read_auto's
    # already-processed fallback to be exercised.
    monkeypatch.setattr(bruker_io, "_read_bruker_binary", fake_read)
    monkeypatch.setattr("opennmr.gui.main_window.QMessageBox.information", lambda *a, **kw: None)

    window = MainWindow()
    window._load(path, vendor="bruker")

    assert window.spectrum_list.count() == 1
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    assert "read_2d_shape_guess_failed" in spectrum.metadata.raw
    assert window._fid_reprocessable(spectrum) is False
    assert window.fid_processing_box.isHidden() is True


def test_main_window_reprocess_fid_works_for_varian(qapp, tmp_path):
    # Regression test: before _fid_reprocessable recognised "varian",
    # this control was simply hidden; now that it's shown, a live
    # control change must actually succeed (varian.read_1d/read_2d
    # take the same fid_shift/align_echo/linear_predict kwargs the GUI
    # always sends).
    window = MainWindow()
    window._load(_make_varian_dataset(tmp_path, n=8), vendor="varian")
    index = window._selected_index()
    original = window.session.spectra[index].data.copy()

    window.fid_shift_spin.setValue(1.0)  # must not raise

    assert not np.allclose(window.session.spectra[index].data, original)  # fid_shift actually took effect


def test_main_window_fid_processing_box_visible_for_jeol(qapp, tmp_path):
    # Regression test: before _fid_reprocessable recognised "jeol"
    # (added alongside xpulse/varian), this box stayed hidden for
    # every JEOL spectrum -- no way to reprocess it from the GUI at all.
    from test_jeol import _write_jdf
    window = MainWindow()
    window._load(_write_jdf(tmp_path), vendor="jeol")
    assert window.fid_processing_box.isHidden() is False


def test_main_window_reprocess_fid_works_for_jeol(qapp, tmp_path):
    # Regression test: jeol.read_1d must accept every kwarg the GUI
    # always sends (window/zero_fill/fid_shift/align_echo/
    # linear_predict_points), including align_echo, which every other
    # vendor's read_1d already accepted but jeol's initially didn't.
    from test_jeol import _write_jdf
    window = MainWindow()
    window._load(_write_jdf(tmp_path, n_points=16), vendor="jeol")

    window.fid_shift_spin.setValue(1.0)
    window.align_echo_button.setChecked(True)  # must not raise


def test_main_window_processing_preview_works_for_varian(qapp, tmp_path):
    # Regression test: before varian.read_1d cached metadata.raw
    # ["fid_raw"], _compute_processing_preview always returned None
    # for a Varian spectrum -- the Processing-preview tab silently
    # never showed anything live for it.
    window = MainWindow()
    window._load(_make_varian_dataset(tmp_path, n=8), vendor="varian")
    spectrum = window.session.spectra[window._selected_index()]
    assert window._compute_processing_preview(spectrum) is not None


def test_main_window_processing_preview_works_for_jeol(qapp, tmp_path):
    from test_jeol import _write_jdf
    window = MainWindow()
    window._load(_write_jdf(tmp_path, n_points=16), vendor="jeol")
    spectrum = window.session.spectra[window._selected_index()]
    assert window._compute_processing_preview(spectrum) is not None


def test_main_window_reprocess_applies_tdeff_truncation(qapp, tmp_path):
    # The stashed "fid" is the fully processed (windowed, zero-filled)
    # FID, so truncation doesn't show up as a shorter array -- it
    # shows up as the truncated tail (originally 0.8, 0.7) reading
    # zero, since truncate happens before zero-fill pads the rest.
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    index = window._selected_index()
    original_fid = window.session.spectra[index].metadata.raw["fid"]
    assert original_fid[2] != 0

    window.tdeff_spin.setValue(2)

    assert window.session.spectra[index].metadata.raw["fid"][2] == 0


def test_main_window_fid_plot_shows_tdeff_marker(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    window.tdeff_spin.setValue(2)
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "TDeff cutoff" in labels


def test_main_window_align_echo_button_immediately_reprocesses_and_shows_fid(qapp, tmp_path):
    # Item G from the user: "Align as full echo" should be an
    # immediate, self-verifying action (a push button), not a
    # checkbox whose effect is only visible after a separate
    # "Reprocess" click with no feedback either way.
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    window.canvas_tabs.setCurrentWidget(window.canvas)  # start on the Spectrum tab

    window.align_echo_button.setChecked(True)

    assert window.canvas_tabs.currentWidget() is window.fid_canvas
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Echo top" in labels


def test_main_window_align_echo_stays_on_processing_preview_when_already_active(qapp, tmp_path):
    # Explicit user feedback: Align Echo was pulling focus out of the
    # Processing-preview tab -- it should stay the default context for
    # processing steps, not force-switch to the plain FID tab when
    # it's already what's showing.
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas

    window.align_echo_button.setChecked(True)

    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas
    fid_ax, _spec_ax = window.processing_preview_canvas.figure.axes
    assert fid_ax.axes.lines  # still drawing something sensible


def test_main_window_echo_top_marker_hidden_when_align_echo_off(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Echo top" not in labels


def test_main_window_align_echo_toggled_off_reverts(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    # Isolate this test to align_echo's own round-trip -- baseline
    # correction is a separate, independent concern (see
    # test_main_window_align_echo_rolls_the_raw_fid's own docstring
    # for why the two are decoupled now).
    spectrum.metadata.raw["baseline_correct"] = False
    window._commit_processing_preview_as_spectrum(spectrum)  # through the same pipeline as the round-trip below
    original = spectrum.data.copy()

    window.align_echo_button.setChecked(True)
    window.align_echo_button.setChecked(False)
    reverted = window.session.spectra[index].data

    np.testing.assert_allclose(original, reverted)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Echo top" not in labels


def test_main_window_align_echo_rolls_the_raw_fid(qapp, tmp_path):
    # align_echo's own direct FID-domain effect (independent of
    # whatever else -- baseline correction, phase -- may or may not be
    # active): rolls the FID so its own peak sits at t=0. Checked
    # against metadata.raw["fid"] (the FID preview tab's own source,
    # now kept live -- see _commit_processing_preview_as_spectrum),
    # not the FFT'd spectrum, so this can't be confounded by any other
    # step in the pipeline.
    echo_fixture = _XPULSE_FIXTURE.replace(
        "0.000000 1.0\n0.000100 0.9\n0.000200 0.8\n0.000300 0.7",
        "0.000000 0.2\n0.000100 0.9\n0.000200 0.3\n0.000300 0.1",
    )
    assert echo_fixture != _XPULSE_FIXTURE  # the substring above must have actually matched
    window = MainWindow()
    path = tmp_path / "echo.jdx"
    path.write_text(echo_fixture)
    window._load(str(path), vendor="xpulse")
    index = window._selected_index()

    before = window.session.spectra[index].metadata.raw["fid"].copy()
    window.align_echo_button.setChecked(True)
    after = window.session.spectra[index].metadata.raw["fid"]

    assert not np.allclose(before, after)
    assert np.argmax(np.abs(after)) == 0  # the peak now sits at t=0


def test_main_window_processing_preview_mirrors_spectrum_for_conventional_2d(qapp, tmp_path, monkeypatch):
    # Regression test: the Processing tab's own preview used to go
    # entirely blank (MplCanvas.clear()) for any conventional
    # (non-pseudo) 2D, since its own from-the-raw-FID dual FID/
    # spectrum rebuild has no way to represent a whole 2D -- explicit
    # user feedback that "Processing figures disappear when the
    # indirect dimension has been processed" (from clicking Process
    # Indirect Dimension) and that "processing and spectrum should
    # update from the same source." Fixed by mirroring the Spectrum
    # tab's own live contour (the same spectrum object, not a second
    # computation) instead of clearing.
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=32, name="previewmirror2d")
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)

    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    fig = window.processing_preview_canvas.figure
    assert any(ax.collections or ax.lines or ax.images for ax in fig.axes)


def test_main_window_f1_truncate_zero_fill_enabled_only_for_conventional_2d(qapp, tmp_path, monkeypatch):
    # Explicit user feedback: "I want the TD and SI options to appear
    # for both dimensions when any 2D is loaded" -- F1's own controls
    # (unlike F2's tdeff_spin, which is X-Pulse-only) must be enabled
    # for a genuine 2D from *any* vendor, and disabled for 1D/pseudo-2D
    # (where "F1" is a slice index, not a frequency dimension with its
    # own FFT to truncate/zero-fill).
    window = MainWindow()

    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="f1gate1d"), vendor="bruker")
    assert window.tdeff_f1_spin.isEnabled() is False
    assert window.zero_fill_f1_spin.isEnabled() is False

    window._load(
        _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="f1gatepseudo2d", f1_nuc="off"),
        vendor="bruker",
    )
    assert window.tdeff_f1_spin.isEnabled() is False
    assert window.zero_fill_f1_spin.isEnabled() is False

    window._load(
        _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="f1gate2d", f1_nuc="13C"),
        vendor="bruker",
    )
    # A fresh 2D load always defaults to pseudo-2D treatment regardless
    # of vendor auto-detection (see _load's own comment) -- Process
    # Indirect Dimension is how a genuine 2D-correlation experiment
    # opts back into a real F1 FFT, same as
    # test_main_window_processing_preview_mirrors_spectrum_for_conventional_2d
    # already relies on.
    window.process_indirect_button.setChecked(True)
    assert window.tdeff_f1_spin.isEnabled() is True
    assert window.zero_fill_f1_spin.isEnabled() is True

    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=4, n2=16, name="f1gatexpulse2d.jdx"), vendor="xpulse")
    assert window.tdeff_f1_spin.isEnabled() is True
    assert window.zero_fill_f1_spin.isEnabled() is True


def test_main_window_f1_truncate_and_zero_fill_change_row_count_live_bruker(qapp, tmp_path, monkeypatch):
    # Truncate alone doesn't shrink the final row count -- a separately
    # -recorded zero-fill target still applies afterward, same
    # truncate/zero-fill independence already established for F2 (and
    # at the io level, see test_bruker_raw_2d.py) -- so both must be
    # set together here to observe the row count actually change.
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="f1livebruker", f1_nuc="13C")
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D -- see gating test's own comment
    index = window._selected_index()
    assert window.session.spectra[index].data.shape[0] == 8

    window.tdeff_f1_spin.setValue(3)
    window.zero_fill_f1_spin.setValue(3)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute

    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[0] == 3
    assert len(spectrum.axes[0].ppm) == 3
    assert len(spectrum.axes[0].hz) == 3


def test_main_window_f1_truncate_row_pair_aware_matches_read_2d_directly(qapp, tmp_path, monkeypatch):
    # The live in-memory recompute path (_recompute_multi_row_data)
    # must treat FnMODE row-pairing the same way io.bruker.read_2d's
    # own truncate_f1/zero_fill_f1 does -- cross-check the two paths
    # against each other on the same synthetic States-TPPI-style
    # (FnMODE 6) raw FID rather than just checking a row count, so a
    # bug that truncates/zero-fills but separates a real row from its
    # imaginary pair (or lands on the wrong axis) wouldn't slip through
    # a shape-only assertion.
    from opennmr.io import bruker as bruker_io

    window = MainWindow()
    path = _make_raw_2d_dataset(
        tmp_path, monkeypatch, n_f1=8, n_f2=16, name="f1pairbruker", f1_nuc="13C", fn_mode=6,
    )
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D -- see gating test's own comment
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    assert spectrum.metadata.raw["fn_mode"] == 6
    assert spectrum.metadata.raw["fid_raw"].shape[0] == 8  # 4 F1 increments, row-paired

    window.tdeff_f1_spin.setValue(3)
    window.zero_fill_f1_spin.setValue(3)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute
    live_result = window.session.spectra[index]
    assert live_result.data.shape[0] == 3

    direct = bruker_io.read_2d(path, truncate_f1=3, zero_fill_f1=3, phase_mode="recorded")
    np.testing.assert_allclose(live_result.data, direct.data, atol=1e-8)
    np.testing.assert_allclose(live_result.data_imag, direct.data_imag, atol=1e-8)
    np.testing.assert_allclose(live_result.axes[0].ppm, direct.axes[0].ppm)


def test_main_window_processing_preview_applies_echo_antiecho_row_transform(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: _compute_processing_preview reads a
    # single row straight out of the cached fid_raw, but for an Echo-
    # Antiecho (FnMODE 6) dataset that cache is missing the extra
    # even-row conjugate-and-reversal read_2d itself applies before F2
    # processing (see io.bruker.read_2d's own _CONJUGATE_QUADRATURE_NOTE-
    # adjacent comment, and _recompute_multi_row_data's matching fix,
    # which this method was apparently never updated to match -- "two
    # pipelines quietly diverge"). Confirmed directly against real data
    # before this fix: a ~208% relative difference for slice 1 (row 0,
    # an echo/even row), not a rounding-level one. Real user report:
    # "the phase is 90 offset" -- a rough description of what a
    # conjugated-but-unreversed trace looks like, not a literal phase
    # error.
    #
    # pseudo-2D mode specifically: slice 1 (row 0 of the ACTUAL
    # committed spectrum) is a valid, direct apples-to-apples reference
    # for what the preview *should* show -- unlike genuine 2D, where a
    # processed row is an F1 frequency bin (a mixture across every raw
    # row), not any single raw row's own trace.
    path = _make_raw_2d_dataset(
        tmp_path, monkeypatch, n_f1=8, n_f2=16, name="echoantiechopreview", f1_nuc="13C", fn_mode=6,
    )
    window = MainWindow()
    window._load(path, vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    assert spectrum.is_pseudo_2d  # default -- see _load's own pseudo-2D-by-default policy
    assert spectrum.metadata.raw["fn_mode"] == 6
    # Isolates the transform under test from an unrelated timing detail:
    # a fresh read_2d-built spectrum.data has no baseline correction
    # baked in yet (that only happens via a live recompute), but
    # _compute_processing_preview applies it live on every call -- off
    # here so the comparison below checks only the row transform.
    spectrum.metadata.raw["baseline_correct"] = False

    actual_row0 = spectrum.complex_data[0]  # row 0 is even -- an echo row
    filled, dt, preview_spectrum, title = window._compute_processing_preview(spectrum)

    assert title == "Slice 1"
    np.testing.assert_allclose(preview_spectrum.complex_data, actual_row0, atol=1e-6)


def test_main_window_recompute_applies_f1_window_matching_read_2d(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: _recompute_multi_row_data never
    # applied any window function to F1 at all -- read_2d always does,
    # defaulting to proc2s's own recorded WDW (the GUI has no live
    # override control for F1's window at all, unlike F2's own
    # window_combo, so "recorded" is the only correct choice). Real
    # user report: "I get a flicker as it saves and it seems to reload
    # something old" while phasing a genuine 2D -- confirmed directly
    # against a real COSY dataset (proc2s WDW=4/QSINE): even a live
    # recompute with *no* phase or FID-processing change at all
    # (phase0=phase1=0, every other control untouched) produced data
    # ~80% different from the original, freshly-loaded spectrum, not a
    # rounding-level gap. Isolated to this exact missing step by first
    # confirming F2's own per-row processing already matched read_2d's
    # own primitives exactly.
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="f1windowmiss2d", f1_nuc="13C")
    _write_params(
        tmp_path / "f1windowmiss2d" / "pdata" / "1" / "proc2s",
        {"SW_p": 2000.0, "SI": 4, "SF": 100.6, "WDW": 4, "SSB": 2.0},  # QSINE, a real, non-trivial window
    )
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    spectrum.metadata.raw["baseline_correct"] = False

    original = spectrum.complex_data.copy()
    window._recompute_multi_row_data(spectrum)  # no control changed at all -- should be a pure no-op

    np.testing.assert_allclose(spectrum.complex_data, original, atol=1e-6)


def test_main_window_recompute_reflects_live_phase_drag_not_just_committed(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: this method only ever read the
    # *committed* phase0/phase1 (metadata.raw), never the live,
    # not-yet-committed delta still sitting in the Phase spin boxes
    # mid-drag. Correct once a drag is released and committed, but this
    # method can also fire *during* an active drag: baseline correction
    # is deliberately skipped on every drag tick (too slow to re-run
    # per-tick) and instead deferred to this same debounced recompute,
    # restarted on every tick while baseline_correct is on. If the drag
    # paused for the 200ms debounce window before being released, this
    # method fired mid-drag and rebuilt spectrum.data using only the
    # *old*, pre-drag committed phase -- silently reverting the live-
    # dragged display back to it. Real user report, part of the same
    # "flicker... reload something old" complaint as the F1-window fix
    # above. Confirmed directly on real data: a live 77-degree F2 drag
    # followed by this method firing showed a ~1.35e9 magnitude
    # difference (not a rounding-level gap) before this fix; exact
    # match after. Checked for both dimensions -- the F1 side had the
    # identical gap, via a matching but separate code path.
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="livedragrace2d", f1_nuc="13C")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    spectrum.metadata.raw["baseline_correct"] = False
    window.side_tabs.setCurrentIndex(window._processing_tab_index)

    window.dim_combo.setCurrentIndex(0)
    window.phase0_handle.setValue(77.0)
    window._apply_phase_preview()
    data_during_f2_drag = spectrum.data.copy()
    window._recompute_multi_row_data(spectrum)  # simulates the debounce firing mid-drag, uncommitted
    np.testing.assert_allclose(spectrum.data, data_during_f2_drag, atol=1e-5)

    window.dim_combo.setCurrentIndex(1)
    window.phase0_handle.setValue(45.0)
    window._apply_phase_preview()
    data_during_f1_drag = spectrum.data.copy()
    window._recompute_multi_row_data(spectrum)
    np.testing.assert_allclose(spectrum.data, data_during_f1_drag, atol=1e-5)


def test_main_window_recompute_uses_each_dimensions_own_pivot_not_the_other_ones(qapp, tmp_path, monkeypatch):
    # Real bug, found and fixed: _recompute_multi_row_data reapplies
    # F2's and F1's own committed phase separately, but both reused the
    # *same* live pivot_spin.value() -- meaningful for only whichever
    # dimension the Phase panel happens to be currently focused on. A
    # nonzero phase1 (F2) committed while F1 is the focused dimension
    # (pivot_spin centred on F1's own, unrelated ppm range) got
    # reapplied around the *wrong* pivot on every live recompute
    # afterward (window/zero-fill/etc.) -- a genuinely different,
    # wrong correction, not a cosmetic one (pivot_frac defines where
    # the phase1 ramp is zero). Confirmed directly: with F1's own
    # centre (0.0 ppm) reused as F2's pivot on a fixture where F2's own
    # range doesn't even contain 0, the resulting data was off by
    # ~129% relative to the correct (centre-pivoted) reference; fixed,
    # it matches to floating-point precision.
    #
    # f1_o1 puts F1's own centre at ~100 ppm, well clear of F2's range.
    # It used to be 0 ppm and that was outside F2's range by accident --
    # the fixture's BF1 disagreed with its SF, which implied a 130 kHz
    # vendor reference and pushed F2 out to around -320 ppm. Making the
    # fixture self-consistent brought F2 back to a normal range that
    # contains 0, so the separation is now arranged deliberately.
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=32, name="pivotmix2d",
                                f1_nuc="13C", f1_o1=10060.0)
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    f1_ppm = spectrum.axis_values(0, "ppm")
    f2_ppm = spectrum.axis_values(-1, "ppm")
    f1_centre = (f1_ppm.min() + f1_ppm.max()) / 2
    assert not (f2_ppm.min() <= f1_centre <= f2_ppm.max())  # F1's centre is outside F2's range

    window.phase1_handle.setValue(50.0)  # commit a real F2 phase1 while F2 is focused
    window.dim_combo.setCurrentIndex(1)  # switch focus to F1 -- pivot_spin recentres onto F1's own range
    assert spectrum.metadata.raw.get("phase1") == pytest.approx(50.0)

    window._recompute_multi_row_data(spectrum)

    from opennmr.io import bruker as bruker_io
    ref = bruker_io.read_2d(path, phase0=0.0, phase1=50.0, pivot_frac=0.5, autophase=False, pseudo_2d=False)
    np.testing.assert_allclose(spectrum.complex_data, ref.complex_data, rtol=1e-6)


def test_main_window_processing_preview_uses_xf2_raw_row_for_genuine_2d(qapp, tmp_path, monkeypatch):
    # Real design requirement, stated explicitly by the user: "2D
    # processing should use slice one of the xf2, whether or not the
    # indirect dimension has been processed, not index 1 of the fully
    # processed data." An earlier round of this exact method tried the
    # opposite (matching spectrum.data's own F1 frequency bin instead,
    # see the git history for that round's own now-removed test) --
    # reverted once the user clarified the intended design directly.
    # This fixture gives every raw row a deliberately different F2
    # frequency, so F1 bin 1 (a combination across all rows) is
    # guaranteed to differ sharply from raw row 0 alone -- making it
    # unambiguous which one the preview is actually built from.
    n_f1, n_f2 = 6, 32
    t2 = np.arange(n_f2) / 8000.0
    rows = [
        np.exp(-t2 * 300) * np.exp(2j * np.pi * (1000.0 + 300.0 * k) * t2)
        for k in range(n_f1)
    ]
    fid = np.array(rows, dtype=np.complex128)
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=n_f1, n_f2=n_f2, name="previewgenuine2d", f1_nuc="13C", fid=fid)

    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    assert not spectrum.is_pseudo_2d
    spectrum.metadata.raw["baseline_correct"] = False

    filled, dt, preview_spectrum, title = window._compute_processing_preview(spectrum)

    assert title == "Slice 1"
    # Must NOT match F1 bin 1 (spectrum.data's own row 0) -- these are
    # guaranteed to differ sharply for this fixture, confirming the
    # preview is genuinely built from the raw row (xf2), not spectrum.data.
    assert not np.allclose(preview_spectrum.complex_data, spectrum.complex_data[0], atol=1.0)
    # Must match raw_fid[0] reprocessed through F2's own pipeline alone.
    raw_fid = spectrum.metadata.raw.get("fid_raw")
    reference = window._process_raw_fid_row(spectrum, raw_fid[0], spectrum.metadata.raw.get("fid_dt"))
    _filled_ref, data_ref = reference
    np.testing.assert_allclose(preview_spectrum.complex_data, data_ref, atol=1e-6)


def test_main_window_zero_fill_f1_and_f2_together_matches_read_2d_directly(qapp, tmp_path, monkeypatch):
    # Real user concern: "the SI for both dimensions doesn't screw up
    # anything else" -- changing F2's own zero-fill (the pre-existing
    # zero_fill_combo/zero_fill_custom_spin control) and F1's
    # (zero_fill_f1_spin) *together*, not just one at a time the way
    # every existing zero-fill test already covers. Cross-checks the
    # GUI's live in-memory recompute against a fresh read_2d call given
    # the identical zero_fill/zero_fill_f1 kwargs, on real code (not a
    # shape-only assertion) -- same discriminating pattern as
    # test_main_window_f1_truncate_row_pair_aware_matches_read_2d_directly.
    from opennmr.io import bruker as bruker_io

    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="zfbothbruker", f1_nuc="13C")
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D
    index = window._selected_index()
    original_f2 = window.session.spectra[index].data.shape[1]

    window._set_zero_fill_value(original_f2 * 2)
    window.zero_fill_f1_spin.setValue(8)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)

    live_result = window.session.spectra[index]
    assert live_result.data.shape == (8, original_f2 * 2)

    direct = bruker_io.read_2d(path, zero_fill=original_f2 * 2, zero_fill_f1=8, phase_mode="recorded")
    np.testing.assert_allclose(live_result.data, direct.data, atol=1e-8)
    np.testing.assert_allclose(live_result.data_imag, direct.data_imag, atol=1e-8)
    np.testing.assert_allclose(live_result.axes[0].ppm, direct.axes[0].ppm)
    np.testing.assert_allclose(live_result.axes[1].ppm, direct.axes[1].ppm)

    # And a subsequent phase interaction on this doubly-resized
    # spectrum must not crash (the pivot-slider crash this session
    # already fixed was triggered by exactly this kind of live-resize
    # -- confirm it holds for a *combined* F1+F2 resize too, not just
    # the single-dimension case that fix's own regression test covers).
    window.phase0_handle.setValue(20.0)
    window._on_pivot_slider_changed(300)
    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[1] == len(spectrum.axes[-1].ppm)


def test_main_window_f1_truncate_and_zero_fill_change_row_count_live_xpulse(qapp, tmp_path):
    window = MainWindow()
    path = _make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=16, name="f1livexpulse.jdx")
    window._load(path, vendor="xpulse")
    index = window._selected_index()
    assert window.session.spectra[index].data.shape[0] == 8 // 2

    window.tdeff_f1_spin.setValue(6)
    window.zero_fill_f1_spin.setValue(16)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute

    # The live rebuild must size F1 the same way the reader does, or
    # the two pipelines disagree the moment an F1 control is touched.
    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[0] == 16
    assert len(spectrum.axes[0].ppm) == 16


def test_main_window_f1_truncate_and_zero_fill_persist_across_reselection(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    path_a = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="f1persista", f1_nuc="13C")
    path_b = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="f1persistb", f1_nuc="13C")
    window._load(path_a, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D -- see gating test's own comment
    window._load(path_b, vendor="bruker")
    window.process_indirect_button.setChecked(True)

    window.spectrum_list.setCurrentRow(0)  # back to A
    window.tdeff_f1_spin.setValue(3)
    window.zero_fill_f1_spin.setValue(3)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute

    window.spectrum_list.setCurrentRow(1)  # away to B
    assert window.tdeff_f1_spin.value() == 8
    assert window.zero_fill_f1_spin.value() == 8

    window.spectrum_list.setCurrentRow(0)  # back to A -- restored, not reset
    assert window.tdeff_f1_spin.value() == 3
    assert window.zero_fill_f1_spin.value() == 3
    assert window.session.spectra[0].data.shape[0] == 3


def test_main_window_reset_processing_clears_f1_truncate_and_zero_fill(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="f1resetbruker", f1_nuc="13C")
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D -- see gating test's own comment
    index = window._selected_index()

    window.tdeff_f1_spin.setValue(3)
    window.zero_fill_f1_spin.setValue(3)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute
    assert window.session.spectra[index].data.shape[0] == 3

    window._on_reset_processing_clicked()

    # Reset discards *processing* choices (window/zero-fill/phase/
    # truncate/baseline) but deliberately preserves whatever "Process
    # Indirect Dimension" choice is currently in effect -- that's a
    # structural interpretation (genuine 2D vs. pseudo-2D), not a
    # processing recipe. Real user report, confirmed by direct
    # reproduction against a real EXSY dataset: Reset used to also
    # silently revert this opt-in back to _load's own pseudo-2D-by-
    # default snapshot, on top of discarding the truncate/zero-fill
    # values it was actually meant to discard.
    assert window.session.spectra[index].is_pseudo_2d is False
    assert window.tdeff_f1_spin.isEnabled() is True
    assert window.zero_fill_f1_spin.isEnabled() is True
    assert window.tdeff_f1_spin.value() == 8  # truncate/zero-fill *values* still correctly discarded
    assert window.zero_fill_f1_spin.value() == 8


def test_main_window_reset_preserves_process_indirect_dimension_choice(qapp, tmp_path, monkeypatch):
    # Focused regression test for the fix itself, isolated from the F1
    # truncate/zero-fill concerns the test above also covers. Real
    # user report, reproduced directly against a real EXSY dataset
    # before fixing: Reset was silently flattening a genuine-2D
    # dataset back to pseudo-2D.
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="resetkeepsgenuine2d", f1_nuc="13C")
    window._load(path, vendor="bruker")
    index = window._selected_index()
    assert window.session.spectra[index].is_pseudo_2d is True  # the usual pseudo-2D-by-default

    window.process_indirect_button.setChecked(True)
    assert window.session.spectra[index].is_pseudo_2d is False

    window._on_reset_processing_clicked()

    assert window.session.spectra[index].is_pseudo_2d is False
    assert window.process_indirect_button.isChecked() is True


def test_main_window_reset_still_reverts_to_pseudo_2d_when_never_opted_out(qapp, tmp_path, monkeypatch):
    # The fix above must not change behaviour for the common case --
    # a 2D never explicitly opted into genuine-2D treatment should
    # still land back on (i.e. stay at) the pseudo-2D default after
    # Reset, same as before this fix.
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="resetstillpseudo2d", f1_nuc="13C")
    window._load(path, vendor="bruker")
    index = window._selected_index()
    assert window.session.spectra[index].is_pseudo_2d is True

    window._on_reset_processing_clicked()

    assert window.session.spectra[index].is_pseudo_2d is True
    assert window.process_indirect_button.isChecked() is False


def test_main_window_pivot_marker_clears_when_leaving_processing_tab(qapp, tmp_path):
    # Explicit user feedback: the dashed pivot-point line left over
    # from phasing stayed visible even after navigating away from the
    # Processing (left) tab entirely, e.g. to Data or Analysis --
    # stale state with nothing left to anchor it to.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.phase0_spin.setValue(15.0)
    assert window.canvas._pivot_line is not None

    window.side_tabs.setCurrentIndex(0)  # Data tab
    assert window.canvas._pivot_line is None
    assert window.processing_preview_canvas._pivot_line is None


def test_main_window_align_echo_on_pseudo2d_does_not_break_phasing(qapp, tmp_path, monkeypatch):
    # Regression test: Align Echo on a real (non-imaging) Bruker
    # pseudo-2D series -- rows with no genuine common echo shape, so
    # find_echo_top's own per-row result varies -- used to crash
    # read_2d entirely (np.apply_along_axis, mismatched per-row
    # lengths), which _read_auto's shape-guess fallback then silently
    # masked by loading TopSpin's own processed data instead (no
    # error shown), which has no raw FID to reprocess from and
    # sometimes no imaginary part at all -- "I can no longer phase
    # anything" afterward. Fixed directly in read_2d (a shared
    # truncation cap across every row); this confirms the whole path
    # end to end, not just the io-level fix.
    from opennmr.io import bruker as bruker_io

    n_f1, n_f2 = 8, 64
    expt = tmp_path / "t2expt"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13, "WDW": 0})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 100.6, "WDW": 0})
    (expt / "ser").write_bytes(b"\x00")
    (expt / "acqus").write_bytes(b"\x00")
    (pdata / "2rr").write_bytes(b"\x00" * (n_f1 * n_f2 * 4))
    (pdata / "2ii").write_bytes(b"\x00" * (n_f1 * n_f2 * 4))

    acqus = {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1.0}
    t2 = np.arange(n_f2)
    rows = []
    for row_index in range(n_f1):
        peak_at = 5 + row_index  # a different "echo top" every row -- no real shared echo
        row = np.zeros(n_f2, dtype=np.complex128)
        row[peak_at] = 1.0 + 0.5j
        row += 0.01 * np.exp(-((t2 - peak_at) ** 2) / 8.0)
        rows.append(row)
    fid = np.array(rows)
    dic = {"acqus": acqus, "acqu2s": {"NUC1": "off", "BF1": 400.13, "O1": 0.0}}
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    # read_acqus_file too: read_1d/read_2d take their parameters from it
    # now, rather than from a combined ng.bruker.read call. Without this
    # the load raises on the placeholder acqus these fixtures write, and
    # _load answers with a modal QMessageBox -- which hangs the suite
    # rather than failing it.
    monkeypatch.setattr(bruker_io, "read_acqus_files", lambda path: dic)
    monkeypatch.setattr(bruker_io, "_read_bruker_binary", lambda p, d, shape=None: fid)
    monkeypatch.setattr(bruker_io, "_remove_digital_filter", lambda d, data: data)
    monkeypatch.setattr(bruker_io, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)

    window = MainWindow()
    window._load(str(expt), vendor="bruker")
    index = window._selected_index()

    window.align_echo_button.setChecked(True)

    spectrum = window.session.spectra[index]
    assert spectrum.data_imag is not None
    assert spectrum.metadata.raw.get("read_2d_shape_guess_failed") is None
    assert window._phase_mode == "pseudo2d"

    before = spectrum.data.copy()
    window.phase0_spin.setValue(30.0)
    assert not np.allclose(window.session.spectra[index].data, before)


def test_main_window_f1_phase_persists_across_live_fid_control_change(qapp, tmp_path, monkeypatch):
    # Regression test: dragging the Phase controls with Dimension set
    # to Indirect (F1) on a conventional 2D wrote the correction
    # straight into spectrum.data (never the problem), but nothing
    # ever recorded it anywhere durable -- unlike F2's own phase0/
    # phase1, which read_2d bakes into metadata.raw. The moment the
    # spin boxes got re-zeroed (switching dim, reselecting) and then
    # something reprocessed the spectrum (a live FID-processing control
    # change, now that there's no "Apply Processing" step at all), F1's
    # phase silently reverted to unphased with no way to tell --
    # explicit user feedback that F1 phasing was "impossible". Fixed
    # via metadata.raw["phase0_f1"/"phase1_f1"], folded in by
    # _commit_pending_phase_to_metadata and reapplied by
    # _recompute_multi_row_data after every live recompute, not just a
    # one-off reload.
    window = MainWindow()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=32, name="f1phase2d")
    window._load(path, vendor="bruker")
    index = window._selected_index()
    # Pseudo-2D is the default treatment for any freshly-loaded 2D
    # (see _load's own comment) -- opt into a real F1 FFT the same way
    # a user would, via the Process Indirect Dimension button.
    window.process_indirect_button.setChecked(True)
    assert window.session.spectra[index].metadata.pseudo_2d is False

    original = window.session.spectra[index].data.copy()
    assert window._phase_mode == "2d"

    window.dim_combo.setCurrentIndex(1)  # Indirect (F1)
    assert window._phase_mode == "2d"

    window.phase0_spin.setValue(40.0)
    spectrum = window.session.spectra[index]
    assert not np.allclose(spectrum.data, original)

    # Re-anchoring (e.g. switching away and back) commits the pending
    # delta into metadata.raw and zeroes the spin box -- exactly the
    # state a later live recompute needs to be tested against, since a
    # real user rarely changes another control mid-drag.
    window._capture_phase_reference()
    assert spectrum.metadata.raw.get("phase0_f1") == pytest.approx(40.0)
    assert window.phase0_spin.value() == 0.0
    f1_phased_data = spectrum.data.copy()

    # Any live FID-processing control change on a conventional 2D
    # triggers a full (debounced) recompute -- must not lose F1's own
    # committed phase.
    window.fid_shift_spin.setValue(0.5)
    _settle_live_reprocess(window)

    recomputed = window.session.spectra[index]
    assert recomputed is spectrum  # updated in place, not replaced
    assert not np.allclose(recomputed.data, original)  # fid_shift took effect
    with_f1_phase = recomputed.data.copy()

    # A/B: clear F1's committed phase and recompute again -- if it was
    # genuinely still being applied above, removing it must change the
    # result.
    spectrum.metadata.raw["phase0_f1"] = 0.0
    window._recompute_multi_row_data(spectrum)
    without_f1_phase = spectrum.data
    assert not np.allclose(with_f1_phase, without_f1_phase)


def _states_tppi_row_paired_fid(n_f1=4, n_f2=32, sw1=2000.0, sw2=8000.0, f1_hz=None):
    # Same forward model as test_bruker_raw_2d.py's own
    # _synthetic_hypercomplex_2d_fid -- a States-TPPI-encoded raw
    # row-pair series (2*n_f1 rows) for a known F1 frequency.
    if f1_hz is None:
        f1_hz = 3 * (sw1 / n_f1)
    t2 = np.arange(n_f2) / sw2
    f2signal = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000.0 * t2)
    rows = []
    for k in range(n_f1):
        sign = -1.0 if k % 2 else 1.0
        phase = 2 * np.pi * f1_hz * (k / sw1)
        rows.append(sign * np.cos(phase) * f2signal)
        rows.append(sign * np.sin(phase) * f2signal)
    return np.array(rows, dtype=np.complex128), f1_hz


def test_main_window_live_recompute_redoes_hypercomplex_decode_for_states_tppi(qapp, tmp_path, monkeypatch):
    # Regression test for a real integration gap found while wiring up
    # States/States-TPPI/Echo-Antiecho F1 decoding: read_2d itself
    # decodes correctly at load time, but _recompute_multi_row_data (a
    # *separate* code path, used for every live F2-only control change
    # on a conventional 2D) redoes F1's FFT from fid_raw's own
    # still-undecoded row count -- without also redoing the same
    # hypercomplex decode there, the very first live control change on
    # a States-TPPI dataset would silently revert to the old, wrong
    # behaviour even though the initial load was correct.
    n_f1 = 4
    fid, _ = _states_tppi_row_paired_fid(n_f1=n_f1, n_f2=32)
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=n_f1, n_f2=32, name="statestppi2d", fn_mode=5, fid=fid)
    window = MainWindow()
    window._load(path, vendor="bruker")
    index = window._selected_index()
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[index]
    assert spectrum.metadata.raw["fn_mode"] == 5

    def f1_peak_index(spec):
        projection = np.abs(spec.data + 1j * spec.data_imag).sum(axis=1)
        return int(np.argmax(projection))

    initial_rows = spectrum.data.shape[0]
    assert initial_rows == n_f1  # decoded at load time -- not left at the raw 2*n_f1
    # read_2d's own load-time decode is already covered/validated
    # elsewhere (test_bruker_raw_2d.py, against an exact Hz target) --
    # here, comparing the live-recompute's peak row against this
    # already-correct baseline avoids re-deriving _ppm_hz_axis's own
    # SF/O1 referencing convention by hand for a 4-point fixture.
    initial_peak_index = f1_peak_index(spectrum)

    # A live F2-only control change routes through _recompute_multi_row_data
    window.fid_shift_spin.setValue(0.5)
    _settle_live_reprocess(window)

    recomputed = window.session.spectra[index]
    assert recomputed.data.shape[0] == n_f1  # still decoded, not doubled back to the raw row count
    assert f1_peak_index(recomputed) == initial_peak_index


def test_main_window_reprocess_updates_both_spectrum_and_fid_canvases(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)

    window.tdeff_spin.setValue(2)

    # FID tab was active at the live-update -> refreshed in place, not
    # left showing the pre-change plot (the TDeff marker moves to the
    # newly-set cutoff).
    assert window.canvas_tabs.currentWidget() is window.fid_canvas
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "TDeff cutoff" in labels

    # The Spectrum canvas (not currently visible) is still kept fresh.
    window.canvas_tabs.setCurrentWidget(window.canvas)
    assert len(window.canvas.ax.lines) > 0


# ============================================================
# Compare tab (multi-spectrum stack/overlay)
# ============================================================

def _select_rows(window, rows):
    window.spectrum_list.clearSelection()
    for row in rows:
        window.spectrum_list.item(row).setSelected(True)


def _trace_lines(ax):
    """The plotted data traces on ``ax``, excluding the baseline
    preview overlay.

    That overlay is a real Line2D on the same axes (see
    ``MplCanvas.set_baseline_preview``) and is drawn by default, so a
    bare ``len(ax.lines)`` no longer counts what these assertions mean
    by it: how many *spectra* are drawn."""
    from opennmr.gui.widgets.canvas import MplCanvas

    return [line for line in ax.lines if line.get_gid() != MplCanvas._BASELINE_PREVIEW_GID]


def _compare_line_count(canvas):
    """Total plotted-trace count for a Compare/pseudo-2D-stack canvas
    -- "Stacked" mode (more than one spectrum) uses one subplot per
    spectrum (see MplCanvas.show_compare's own docstring for why:
    a single shared y-axis can't keep every trace's baseline fixed
    under a rescale), so the lines are spread across
    canvas._compare_axes rather than all sitting on canvas.ax alone;
    "Superimposed" mode (and a single-spectrum "stack") still uses
    one shared axes, so this falls back to canvas.ax.lines there."""
    if canvas._compare_axes is not None:
        return sum(len(ax.lines) for ax in canvas._compare_axes)
    return len(canvas.ax.lines)


def test_main_window_spectrum_list_allows_multi_select(qapp, tmp_path):
    from PySide6.QtWidgets import QAbstractItemView
    window = MainWindow()
    assert window.spectrum_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection


def test_main_window_single_selection_does_not_trigger_compare(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0])
    assert window.compare_status_label.text() == "Select at least two 1D spectra to compare."
    # Canvas shows the single spectrum, not a multi-line comparison.
    assert len(_trace_lines(window.canvas.ax)) == 1


def test_main_window_multi_selection_shows_compare(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.compare_status_label.text() == "Comparing 2 spectra."
    assert _compare_line_count(window.compare_canvas) == 2
    assert window.canvas_tabs.currentWidget() is window.compare_canvas


def test_main_window_compare_list_preserves_a_current_row_but_never_preselects(qapp, tmp_path):
    # Explicit user request: "in compare, none of the spectra should be
    # pre-selected like the last one is now." This narrows, rather than
    # reverses, the earlier feedback that it was "no longer obvious
    # which spectrum is selected" -- that was about the highlight being
    # invisible once a row *had* been chosen, which still works below.
    # What's gone is the choice made on the user's behalf before they
    # have clicked anything.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.compare_list.currentRow() == -1

    # Once the user does pick one, it survives a rebuild -- Move
    # Up/Down and check-state toggles must not lose their place.
    window.compare_list.setCurrentRow(0)
    window._on_move_compare_item(1)
    assert window.compare_list.currentRow() == 1
    moved_entry = window.compare_list.item(1).data(Qt.ItemDataRole.UserRole)

    _select_rows(window, [0])
    _select_rows(window, [0, 1])  # a fresh rebuild, same two entries
    assert window.compare_list.item(window.compare_list.currentRow()).data(Qt.ItemDataRole.UserRole) == moved_entry


def test_main_window_compare_list_selection_highlights_matching_plot_row(qapp, tmp_path):
    # Follow-up real bug: compare_list did get a current row (previous
    # test), and MplCanvas already had a click-to-highlight mechanism
    # for a Compare row (_selected_compare_index/_apply_compare_
    # selection_highlight) -- but the two were never connected, and
    # separately, show_compare's own "let peaks bleed into neighbouring
    # rows" fix (row_ax.patch.set_alpha(0.0)) silently made *any*
    # facecolor tint on that patch invisible, including this one, once
    # it *was* wired up. Both needed fixing before a highlight actually
    # rendered anywhere.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="c"), vendor="bruker")
    _select_rows(window, [0, 1, 2])
    # Nothing pre-selected (see the test above), so no row is tinted
    # until the user actually picks one.
    assert window.compare_list.currentRow() == -1
    assert all(ax.patch.get_alpha() == pytest.approx(0.0)
               for ax in window.compare_canvas._compare_axes)

    window.compare_list.setCurrentRow(0)
    assert window.compare_canvas._selected_compare_index == 0
    assert window.compare_canvas._compare_axes[0].patch.get_alpha() == pytest.approx(1.0)
    assert window.compare_canvas._compare_axes[1].patch.get_alpha() == pytest.approx(0.0)

    window.compare_list.setCurrentRow(2)
    assert window.compare_canvas._selected_compare_index == 2
    assert window.compare_canvas._compare_axes[2].patch.get_alpha() == pytest.approx(1.0)
    assert window.compare_canvas._compare_axes[0].patch.get_alpha() == pytest.approx(0.0)


def test_main_window_compare_list_selection_maps_around_unchecked_rows(qapp, tmp_path):
    # compare_list row index and the canvas's own per-row axes index
    # diverge as soon as any row is unchecked (dropped from the plot
    # entirely) -- the mapping between the two must account for that,
    # not just assume row N in the list is row N in the plot.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="c"), vendor="bruker")
    _select_rows(window, [0, 1, 2])

    window.compare_list.item(0).setCheckState(Qt.CheckState.Unchecked)  # drops the first plotted row
    window.compare_list.setCurrentRow(2)  # third list row, but second *plotted* row now
    assert window.compare_canvas._selected_compare_index == 1

    window.compare_list.setCurrentRow(0)  # current row is now the unchecked one -- nothing plotted there
    assert window.compare_canvas._selected_compare_index is None


def test_main_window_deselecting_back_to_single_restores_single_view(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert _compare_line_count(window.compare_canvas) == 2

    _select_rows(window, [0])
    assert len(_trace_lines(window.canvas.ax)) == 1
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_compare_skips_non_1d_spectra(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_2d_dataset(tmp_path, name="b2d"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert "not 1D" in window.compare_status_label.text()
    assert "Select at least two" in window.compare_status_label.text()


def test_main_window_compare_overlays_multiple_2d_spectra(qapp, tmp_path):
    # Real roadmap ask: "a way to overlay multiple 2D contours."
    # plotting.twod.overlay_plot already existed (ported from
    # nmrtools2's own NMR2D_overlay_plot), just never wired to the GUI.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, name="a2d"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    window._load(_make_2d_dataset(tmp_path, name="b2d"), vendor="bruker")
    window.process_indirect_button.setChecked(True)

    _select_rows(window, [0, 1])

    assert "Overlaying 2 2D spectra" in window.compare_status_label.text()
    assert window.canvas_tabs.currentWidget() is window.compare_canvas
    assert window.compare_canvas._contour_spectrum is None  # not a single-spectrum contour view
    assert len(window.compare_canvas.ax.collections) >= 2  # one contour-line collection per spectrum


def test_main_window_compare_2d_overlay_rejects_incompatible_spectra(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, name="het", f1_nuc="13C"), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    window._load(_make_2d_dataset(tmp_path, name="homo", f1_nuc="1H"), vendor="bruker")
    window.process_indirect_button.setChecked(True)

    _select_rows(window, [0, 1])

    # overlay_plot's own compatibility check (different F1 nuclides)
    # surfaces as a status message, not a crash -- same "quiet status
    # text, no popup, no forced tab switch" treatment the existing "not
    # 1D"/"no sample mass set" skip-reasons already get, since this is
    # an expected/reportable mismatch, not an exceptional failure.
    assert "nuclide" in window.compare_status_label.text().lower()


def test_main_window_compare_2d_overlay_takes_up_to_six_spectra(qapp, tmp_path):
    # The limit was 2-3 (the notebook's own); raised to 6 on explicit
    # request. Past that, the spectra are shown side by side, and the
    # status line says why, rather than nothing being drawn.
    window = MainWindow()
    names = [f"s{i}" for i in range(7)]
    for name in names:
        window._load(_make_2d_dataset(tmp_path, name=name), vendor="bruker")
        window.process_indirect_button.setChecked(True)

    _select_rows(window, [0, 1, 2, 3])
    assert "Overlaying 4 2D spectra" in window.compare_status_label.text()

    _select_rows(window, list(range(7)))
    text = window.compare_status_label.text()
    assert "Can't overlay" in text and "side by side" in text
    assert len(window.compare_canvas._panel_axes) == 7


def test_main_window_compare_superimposed_layout_runs(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])

    window.layout_mode_combo.setCurrentIndex(1)  # Superimposed
    assert len(window.compare_canvas.ax.lines) == 2


def test_main_window_compare_stays_primed_when_the_experiment_changes(qapp, tmp_path):
    # Reported directly: "the compare tab should be primed and ready to
    # go and not need to be turned off and on again (switch between
    # superimposed and stacked and then back again)".
    #
    # Selecting another pseudo-2D while Compare was the visible tab
    # force-switched to Spectrum -- because the "is Compare already
    # showing this spectrum's stack?" check ran while the list still
    # held the *previous* spectrum's slices. Compare was then left
    # unrendered, since its refresh defers the draw whenever it isn't
    # the visible tab.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path / "c", n_f1=4, n_f2=32), vendor="bruker")
    window._load(_make_2d_dataset(tmp_path / "d", n_f1=6, n_f2=32), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    assert len(window.compare_canvas._compare_axes) == 6

    window._select_single_row(0)

    # Still on Compare, and already showing the newly selected
    # spectrum's own slices -- no mode toggle needed to wake it up.
    assert window.canvas_tabs.currentWidget() is window.compare_canvas
    assert window._compare_view_stale is False
    assert len(window.compare_canvas._compare_axes) == 4

    window._select_single_row(1)
    assert window.canvas_tabs.currentWidget() is window.compare_canvas
    assert len(window.compare_canvas._compare_axes) == 6


def test_main_window_selecting_a_1d_still_leaves_the_compare_tab(qapp, tmp_path):
    # The other half of the same rule: a plain 1D has no slice stack to
    # put in Compare, so staying there would leave a blank panel up.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path / "e", n_f1=4, n_f2=32), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="plain"), vendor="bruker")
    window._select_single_row(0)
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)

    window._select_single_row(1)
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_click_modes_reset_when_the_experiment_changes(qapp, tmp_path):
    # Reported directly: "a lot of specific buttons for example stay
    # selected after switching experiment file". Pick Peak and Select
    # Region arm the Spectrum canvas to treat the next click as
    # something other than a click -- left latched across a selection
    # change, that click lands on the wrong dataset.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="one"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="two"), vendor="bruker")

    # One mode at a time (they are mutually exclusive since
    # 2026-09-23), so each is armed on its own and must be reset by the
    # experiment change.
    for arm, armed in (
        (window.pick_peak_button, lambda: window.canvas._pick_mode),
        (window.select_region_button, lambda: window.canvas._region_select_mode),
        (window.measurement_menu_action, lambda: window.canvas._measurement_mode),
    ):
        window._select_single_row(1)
        arm.setChecked(True)
        assert armed() is True

        window._select_single_row(0)

        assert arm.isChecked() is False
        # The canvas is disarmed too, not just the button un-drawn.
        assert armed() is False


def test_main_window_per_spectrum_settings_survive_the_experiment_change(qapp, tmp_path):
    # The reset above is for unfinished gestures only -- it must not
    # take saved per-spectrum work with it.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="keep1"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="keep2"), vendor="bruker")
    window._select_single_row(0)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._add_region_row(4.0, 6.0)
    n_peaks = window.peak_table.rowCount()
    n_regions = window.region_table.rowCount()

    window._select_single_row(1)
    window._select_single_row(0)

    assert window.peak_table.rowCount() == n_peaks
    assert window.region_table.rowCount() == n_regions


def test_main_window_compare_normalize_by_mass_skips_unset(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window.session.set_mass(0, 2.0)  # only spectrum 0 has a mass set

    _select_rows(window, [0, 1])
    window.compare_normalize_combo.setCurrentIndex(2)  # By sample mass

    assert "no sample mass set" in window.compare_status_label.text()
    assert "Select at least two" in window.compare_status_label.text()


def test_main_window_compare_normalize_by_mass_scales_data(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, n=64, name="b"), vendor="bruker")
    window.session.set_mass(0, 2.0)
    window.session.set_mass(1, 4.0)

    _select_rows(window, [0, 1])
    window.compare_normalize_combo.setCurrentIndex(2)  # By sample mass

    assert window.compare_status_label.text() == "Comparing 2 spectra."
    original_a = window.session.spectra[0].data.copy()
    # "Stacked" mode is one subplot per spectrum, in order -- the
    # first spectrum's own trace is the first subplot's line, not
    # necessarily canvas.ax (the bottom/last subplot).
    plotted_a = window.compare_canvas._compare_axes[0].lines[0].get_ydata()
    # Plotted line reflects the /mass=2.0 scaling...
    assert not np.allclose(original_a, plotted_a)
    np.testing.assert_allclose(plotted_a * 2.0, original_a)
    # ...but the session's own stored data is untouched (comparison
    # copies are ephemeral, never written back).
    np.testing.assert_allclose(original_a, window.session.spectra[0].data)


def test_main_window_set_mass_action_via_dialog(qapp, tmp_path, monkeypatch):
    # Entered in mg (typical sample sizes are tens of mg), stored in
    # metadata.mass as grams -- see _on_set_mass_action's own comment.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (35.0, True))
    window._on_set_mass_action(0)

    assert window.session.spectra[0].metadata.mass == pytest.approx(0.035)
    assert "Sample mass" in window.info_label.text()
    assert "35" in window.info_label.text()  # displayed back in mg, not 0.035 g


def test_main_window_set_mass_action_cancelled_is_noop(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (3.5, False))
    window._on_set_mass_action(0)

    assert window.session.spectra[0].metadata.mass is None


def test_main_window_compare_list_populated_from_multi_selection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])

    assert window.compare_list.count() == 2
    entries = [window.compare_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(2)]
    assert sorted(entries) == [(0, None), (1, None)]  # whole spectra, not slices
    for i in range(2):
        assert window.compare_list.item(i).checkState() == Qt.CheckState.Checked


def test_main_window_unchecking_compare_item_drops_it_from_plot(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="c"), vendor="bruker")
    _select_rows(window, [0, 1, 2])
    assert window.compare_status_label.text() == "Comparing 3 spectra."

    window.compare_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert window.compare_status_label.text() == "Comparing 2 spectra."


def test_main_window_move_compare_item_reorders_local_plot_only(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    original_session_order = list(window.session.spectra)

    window.compare_list.setCurrentRow(0)
    window._on_move_compare_item(1)

    # Local compare-list order swapped...
    assert window.compare_list.item(0).data(Qt.ItemDataRole.UserRole) == (1, None)
    assert window.compare_list.item(1).data(Qt.ItemDataRole.UserRole) == (0, None)
    # ...but the master session order is untouched.
    assert list(window.session.spectra) == original_session_order


def test_main_window_move_compare_item_out_of_bounds_is_noop(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])

    window.compare_list.setCurrentRow(0)
    window._on_move_compare_item(-1)  # already first -- no row above
    assert window.compare_list.item(0).data(Qt.ItemDataRole.UserRole) == (0, None)


def test_main_window_compare_list_preserves_unchecked_state_across_reselect(qapp, tmp_path):
    # Ctrl-clicking one more spectrum in the Data tab shouldn't wipe
    # out a deselection already made in the Compare tab.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="c"), vendor="bruker")
    _select_rows(window, [0, 1])
    window.compare_list.item(0).setCheckState(Qt.CheckState.Unchecked)

    _select_rows(window, [0, 1, 2])  # add spectrum c to the selection

    checked = {
        window.compare_list.item(i).data(Qt.ItemDataRole.UserRole): window.compare_list.item(i).checkState()
        for i in range(window.compare_list.count())
    }
    assert checked[(0, None)] == Qt.CheckState.Unchecked
    assert checked[(1, None)] == Qt.CheckState.Checked
    assert checked[(2, None)] == Qt.CheckState.Checked


def test_main_window_removing_spectrum_rebuilds_compare_list(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="c"), vendor="bruker")
    _select_rows(window, [1, 2])  # b, c

    # Remove "a" alone (the right-click ▸ Remove path), leaving b and c
    # selected and in the Compare list. (Remove removes every *selected*
    # spectrum since 2026-09-30, so it cannot be used for this.)
    window._remove_rows([0])

    # b/c shifted down to indices 0/1 -- the compare list must reflect
    # that, not stale indices 1/2 (now out of range or wrong spectra).
    entries = {window.compare_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(window.compare_list.count())}
    assert entries == {(0, None), (1, None)}


def test_main_window_clear_resets_compare_status(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.compare_status_label.text() == "Comparing 2 spectra."

    window._on_clear_clicked()
    assert window.compare_status_label.text() == "Select at least two 1D spectra to compare."


# ============================================================
# Item D: Compare auto-populated from a pseudo-2D's own slices
# ============================================================

def test_main_window_compare_auto_populates_from_pseudo2d_slices(qapp):
    delays = np.geomspace(0.001, 0.3, 5)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)  # a single selection -- no ctrl-click needed

    assert window.compare_list.count() == 5
    entries = [window.compare_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(5)]
    assert entries == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]  # in slice order
    for i in range(5):
        assert window.compare_list.item(i).checkState() == Qt.CheckState.Checked
    assert window.compare_status_label.text() == "Comparing 5 spectra."
    # canvas_tabs itself should land on Spectrum, not Compare --
    # explicit user feedback that loading/selecting a pseudo-2D
    # shouldn't jump to Compare just because its slices got listed
    # there as a side effect.
    assert window.canvas_tabs.currentWidget() is window.canvas
    # The actual Compare canvas render is deferred until the tab is
    # genuinely visited (see _refresh_compare_view's own docstring --
    # rendering hundreds/thousands of subplots nobody is looking at,
    # for a pseudo-2D series with many slices, measured at ~25s on
    # real data) -- confirm it's marked stale, then visiting Compare
    # actually draws it.
    assert window._compare_view_stale is True
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    assert _compare_line_count(window.compare_canvas) == 5


def test_main_window_compare_not_populated_for_single_1d_selection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.compare_list.count() == 0


def test_main_window_compare_list_context_menu_requires_checked_entry(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    window.compare_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.compare_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    import opennmr.gui.main_window as main_window_module
    captured = []
    monkeypatch.setattr(main_window_module, "QMenu", lambda *a, **k: captured.append(_FakeMenu()) or captured[-1])

    window._on_compare_list_context_menu(window.compare_list.rect().center())

    assert captured == []  # menu never even built -- nothing checked to fit


def test_main_window_compare_list_context_menu_offers_fit_curve(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])

    import opennmr.gui.main_window as main_window_module
    captured = []
    monkeypatch.setattr(main_window_module, "QMenu", lambda *a, **k: captured.append(_FakeMenu()) or captured[-1])

    window._on_compare_list_context_menu(window.compare_list.rect().center())

    assert any("Fit Curve" in label for label in captured[0].added_labels)


def test_main_window_fit_curve_compare_stack_recovers_known_t2(qapp):
    # Item C+D together: a pseudo-2D's auto-populated slices, curve-fit
    # via the Compare tab's own context menu -- not the pseudo-2D
    # spectrum-list path -- must recover the same known T2.
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 12)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)  # also auto-populates compare_list
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"

    window._on_fit_curve_compare_stack_action()

    assert "T2" in window.curve_fit_summary_label.text()
    assert len(window.fit_canvas.ax.lines) == 2
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_main_window_fit_curve_custom_equation_recovers_known_t2(qapp):
    # User-typed sympy equation, selected via "Custom equation…" in the
    # Model combo, should fit exactly like a built-in model -- here
    # A*exp(-x/T2) is algebraically the same curve exponential_decay
    # already fits, so the recovered T2 should match just as well.
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 12)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    custom_index = window.relaxation_model_combo.findData("custom")
    window.relaxation_model_combo.setCurrentIndex(custom_index)
    window.custom_equation_edit.setText("A*exp(-x/T2)")
    window.custom_p0_edit.setText("1.0, 0.03")

    window._on_fit_curve_button_clicked()

    result = window._relaxation_fit_results[0]
    assert result["param_names"] == ["A", "T2"]
    assert result["params"]["T2"] == pytest.approx(true_t2, rel=0.1)
    # The report names every parameter and says what its ± means -- see
    # processing.relaxation.format_relaxation_report.
    report = window.curve_fit_summary_label.text()
    assert "A" in report and "T2" in report
    assert "Uncertainty" in report and "1σ" in report
    assert "R²" in report


def test_main_window_custom_equation_fields_hidden_unless_custom_model_selected(qapp):
    window = MainWindow()
    assert window.custom_equation_edit.isHidden() is True

    custom_index = window.relaxation_model_combo.findData("custom")
    window.relaxation_model_combo.setCurrentIndex(custom_index)
    assert window.custom_equation_edit.isHidden() is False

    window.relaxation_model_combo.setCurrentIndex(0)
    assert window.custom_equation_edit.isHidden() is True


def test_main_window_custom_equation_live_preview_shows_detected_parameters(qapp):
    window = MainWindow()
    custom_index = window.relaxation_model_combo.findData("custom")
    window.relaxation_model_combo.setCurrentIndex(custom_index)

    window.custom_equation_edit.setText("A*exp(-x/T1)+C")
    assert window.custom_params_label.text() == "Parameters (in order): A, T1, C"

    window.custom_equation_edit.setText("A*exp(-x/T1")  # unbalanced parens
    assert window.custom_params_label.text().startswith("Error:")


def test_main_window_fitting_menu_model_submenu_tracks_fit_type(qapp):
    # Explicit design: the top menu bar's Analysis/Fitting menus are
    # quick access to controls that already exist in the Analysis tab
    # (not new capability) -- picking a fit type here must drive the
    # same combo the Curve Fitting sub-tab uses, model list included.
    from opennmr.gui.main_window import _DIFFUSION_MODEL_ITEMS

    window = MainWindow()
    fitting_menu = next(
        a for a in window.menuBar().actions() if a.text().replace("&", "") == "Fitting"
    ).menu()
    fit_type_submenu = next(
        a for a in fitting_menu.actions() if a.menu() and "Type" in a.text().replace("&", "")
    ).menu()

    fit_type_submenu.actions()[1].trigger()  # "Diffusion (PFG)"
    assert window.fit_type_combo.currentIndex() == 1

    window._sync_fitting_menu_checks()
    assert [a.text() for a in window._model_menu.actions()] == [
        label for label, _key in _DIFFUSION_MODEL_ITEMS
    ] + ["Custom equation…"]
    assert fit_type_submenu.actions()[1].isChecked() is True


def test_main_window_view_relaxation_peak_spin_disabled_before_fit(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.view_relaxation_peak_spin.isEnabled() is False


def test_main_window_view_relaxation_peak_spin_switches_between_peaks(qapp):
    # Explicit user request: only one peak's curve fit shown at a
    # time, with a way to switch between them -- previously every
    # peak was plotted together on one axes.
    delays = np.geomspace(0.001, 0.3, 10)
    ppm = np.linspace(3.0, 7.0, 800)
    rng = np.random.default_rng(0)
    t2_a, t2_b = 0.02, 0.08
    data = np.zeros((len(delays), len(ppm)))
    for i, t in enumerate(delays):
        data[i] = (
            gaussian(ppm, 5.0, 0.05, np.exp(-t / t2_a))
            + gaussian(ppm, 6.0, 0.05, np.exp(-t / t2_b))
            + rng.normal(0, 0.002, len(ppm))
        )
    f1_axis = NMRAxis(kind="index", index=np.arange(1, len(delays) + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    spectrum = NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._add_peak_row(shift=6.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"

    window._on_fit_curve_button_clicked()

    assert window.view_relaxation_peak_spin.isEnabled() is True
    assert window.view_relaxation_peak_spin.maximum() == 2
    assert window.view_relaxation_peak_spin.value() == 1
    label_peak1 = window.relaxation_fit_params_label.text()

    window.view_relaxation_peak_spin.setValue(2)
    label_peak2 = window.relaxation_fit_params_label.text()

    assert label_peak1 != label_peak2
    # Only one peak's data+fit ever on screen at once -- 2 lines
    # (fit + data), never 4 (both peaks' worth).
    assert len(window.fit_canvas.ax.lines) == 2

    # Explicit user request: arrows like the phasing-tab slice picker.
    window.view_relaxation_peak_spin.setValue(1)
    window.relaxation_peak_next_button.click()
    assert window.view_relaxation_peak_spin.value() == 2
    window.relaxation_peak_next_button.click()  # past the last peak -- must clamp, not raise
    assert window.view_relaxation_peak_spin.value() == 2
    window.relaxation_peak_prev_button.click()
    assert window.view_relaxation_peak_spin.value() == 1
    window.relaxation_peak_prev_button.click()  # past the first peak -- must clamp, not raise
    assert window.view_relaxation_peak_spin.value() == 1


def test_main_window_relaxation_uncertainty_checkbox_shows_shaded_band(qapp):
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 12)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)

    window._on_fit_curve_button_clicked()
    assert len(window.fit_canvas.ax.collections) == 0  # no band by default

    window.relaxation_uncertainty_checkbox.setChecked(True)
    assert len(window.fit_canvas.ax.collections) == 1  # fill_between's PolyCollection

    window.relaxation_uncertainty_checkbox.setChecked(False)
    assert len(window.fit_canvas.ax.collections) == 0


def _fit_a_t2_curve(window, delays, true_t2, noise=0.0, seed=0):
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2, noise=noise, seed=seed)
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    # These delays are in seconds; the panel's own default is ms.
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"
    window._on_fit_curve_button_clicked()


def test_main_window_exclude_nearest_point_removes_it_and_refits(qapp):
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    index = 0
    original = window._relaxation_fit_original[index]
    n_before = len(original["t"])

    # Right-click exactly on the first real data point.
    x, y = float(original["t"][0]), float(original["y"][0])
    window._on_fit_point_exclude_requested(x, y)

    assert window._excluded_relaxation_indices.get(index) == {0}
    refit = window._relaxation_fit_results[index]
    assert len(refit["t"]) == n_before - 1
    assert x not in refit["t"]
    # The original (pristine) result is untouched -- Reset can still
    # get back to it.
    assert len(window._relaxation_fit_original[index]["t"]) == n_before


def test_main_window_exclude_keeps_the_starting_values_the_fit_used(qapp):
    # Reported directly: "when fit points are excluded, the parameters
    # set in starting values get ignored". The refit fell back to
    # fit_relaxation's own automatic p0/bounds, on a since-outdated
    # assumption that the GUI had no way to set them.
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("exponential_decay"))

    window.guess_group.setChecked(True)
    window.guess_table.setItem(1, 1, QTableWidgetItem("0.04"))   # T2 guess
    window.guess_table.setItem(1, 2, QTableWidgetItem("0.02"))   # T2 min
    window.guess_table.setItem(1, 3, QTableWidgetItem("0.09"))   # T2 max
    window._on_fit_curve_button_clicked()

    original = window._relaxation_fit_original[0]
    window._on_fit_point_exclude_requested(float(original["t"][0]), float(original["y"][0]))

    refit = window._relaxation_fit_results[0]
    assert len(refit["t"]) == len(original["t"]) - 1      # the point really went
    # The bounds the panel asked for are the bounds it was refitted in.
    assert refit["bounds"][0][1] == pytest.approx(0.02)
    assert refit["bounds"][1][1] == pytest.approx(0.09)
    assert refit["p0"][1] == pytest.approx(0.04)


def test_main_window_exclude_works_on_a_multicomponent_diffusion_fit(qapp):
    # The refit passed model="diffusion_n" with no spec to go with it,
    # which fit_relaxation rejects outright -- so excluding a point
    # from a multicomponent fit could not work at all.
    b = np.linspace(0.0, 3.0e9, 12)
    ppm = np.linspace(0.0, 10.0, 200)
    data = np.array([
        lorentzian(ppm, 5.0, 0.08, 1.0) * (0.6 * np.exp(-2.0e-9 * bi) + 0.4 * np.exp(-2.0e-10 * bi))
        for bi in b
    ])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, b.size + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.08, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)                 # Diffusion
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{v:.8g}" for v in b))
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("diffusion_n"))
    window.diffusion_components_spin.setValue(2)
    window._on_fit_curve_button_clicked()
    assert window._relaxation_fit_results

    original = window._relaxation_fit_original[0]
    n_before = len(original["t"])
    window._on_fit_point_exclude_requested(float(original["t"][-1]), float(original["y"][-1]))

    assert len(window._relaxation_fit_results[0]["t"]) == n_before - 1


def test_main_window_exclude_marker_follows_the_spectrum_it_belongs_to(qapp):
    # Selecting another spectrum swapped in its own siloed fit results
    # but left the exclusion machinery pointing at the previous one's
    # series -- so the next right-click refitted the wrong data.
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    original = window._relaxation_fit_original[0]
    window._on_fit_point_exclude_requested(float(original["t"][0]), float(original["y"][0]))
    assert window._excluded_relaxation_indices.get(0) == {0}

    other = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.02)
    _load_spectrum_directly(window, other)

    # The new selection carries no fit and no exclusions of its own.
    assert window._excluded_relaxation_indices == {}
    assert window._relaxation_fit_original is None or not window._relaxation_fit_original


def test_main_window_exclude_nearest_point_shows_excluded_marker(qapp):
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    original = window._relaxation_fit_original[0]

    window._on_fit_point_exclude_requested(float(original["t"][0]), float(original["y"][0]))

    labels = [line.get_label() for line in window.fit_canvas.ax.lines]
    assert "Excluded" in labels


def test_main_window_reset_excluded_points_restores_all(qapp):
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    index = 0
    original = window._relaxation_fit_original[index]
    n_before = len(original["t"])

    window._on_fit_point_exclude_requested(float(original["t"][0]), float(original["y"][0]))
    assert len(window._relaxation_fit_results[index]["t"]) == n_before - 1

    window._on_fit_points_reset_requested()

    assert window._excluded_relaxation_indices.get(index, set()) == set()
    assert len(window._relaxation_fit_results[index]["t"]) == n_before


def test_main_window_exclude_point_refuses_below_two_remaining(qapp, monkeypatch):
    monkeypatch.setattr("opennmr.gui.main_window.QMessageBox.information", lambda *a, **kw: None)
    delays = np.array([0.01, 0.1])  # exactly 2 points -- the minimum
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    original = window._relaxation_fit_original[0]

    window._on_fit_point_exclude_requested(float(original["t"][0]), float(original["y"][0]))

    # Refused (would leave only 1 point) -- nothing actually excluded,
    # nothing about the displayed fit changed.
    assert window._excluded_relaxation_indices.get(0, set()) == set()
    assert len(window._relaxation_fit_results[0]["t"]) == 2


def test_main_window_exclude_point_noop_with_no_fit(qapp):
    window = MainWindow()
    window._on_fit_point_exclude_requested(0.1, 0.5)  # must not raise
    window._on_fit_points_reset_requested()  # must not raise


def test_main_window_fit_curve_compare_stack_requires_two_spectra(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._add_peak_row(shift=5.0, fwhm=1.0, eta=0.5)

    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))
    window._on_fit_curve_compare_stack_action()  # compare_list is empty -- nothing checked
    assert shown.get("called") is True


# ============================================================
# Analysis tab (pseudo-Voigt peak fitting -> relaxation)
# ============================================================

def _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05, M0=1.0,
                                        n_pts=800, noise=0.002, seed=0):
    """A synthetic pseudo-2D relaxation series with a known, exactly
    recoverable T2 -- one Gaussian peak whose amplitude decays
    ``M0 * exp(-t/T2)`` across slices."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    rng = np.random.default_rng(seed)
    data = np.zeros((len(delays), n_pts))
    for i, t in enumerate(delays):
        amplitude = M0 * np.exp(-t / T2)
        data[i] = gaussian(ppm, x0, fwhm, amplitude) + rng.normal(0, noise, n_pts)

    f1_axis = NMRAxis(kind="index", index=np.arange(1, len(delays) + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


def _isolate_sidecars(monkeypatch, tmp_path):
    """Point sidecar reads/writes at a scratch directory, keyed per
    source path. Per-path, not one shared file: a single shared sidecar
    leaks one dataset's saved state into the next one's load, which
    silently invalidates any before/after comparison across datasets."""
    import hashlib

    def _path(source_path, suffix):
        key = hashlib.sha1(str(source_path).encode()).hexdigest()[:16]
        return str(tmp_path / f"{key}_{suffix}.json")

    monkeypatch.setattr(io_persistence, "metadata_sidecar_path", lambda p: _path(p, "meta"))
    monkeypatch.setattr(io_persistence, "processing_sidecar_path", lambda p: _path(p, "proc"))


def _load_spectrum_directly(window, spectrum):
    """Append an already-constructed NMRSpectrum straight into the
    session, bypassing file I/O -- for synthetic data no importer
    produces."""
    window.session.spectra.append(spectrum)
    window.session._load_info.append({"path": "synthetic", "vendor": "bruker", "kwargs": {}})
    window.spectrum_list.addItem(window.session.label(len(window.session) - 1))
    window._select_single_row(len(window.session) - 1)


def _settle_canvas_2d_refresh(window):
    """:func:`_settle_live_reprocess`'s own sibling for
    :attr:`MainWindow._canvas_2d_refresh_timer` -- force a pending
    debounced 2D/pseudo-2D contour redraw on ``self.canvas`` to run
    immediately, instead of a real test waiting out the debounce."""
    if window._canvas_2d_refresh_timer.isActive():
        window._canvas_2d_refresh_timer.stop()
        window._run_debounced_canvas_2d_refresh()


def _settle_live_reprocess(window):
    """Force a pending debounced multi-row recompute
    (:attr:`MainWindow._live_reprocess_timer`, see its own docstring)
    to run immediately, instead of a real test waiting out the debounce
    interval -- multi-row baseline correction, and any FID-processing
    control change on a pseudo-2D "apply to every slice"/conventional
    2D spectrum, schedule one rather than recomputing inline. A no-op
    if nothing is actually pending."""
    if window._live_reprocess_timer.isActive():
        window._live_reprocess_timer.stop()
        window._run_debounced_multi_row_reprocess()


def _settle_sidecar_save(window):
    """:func:`_settle_live_reprocess`'s own sibling for
    :attr:`MainWindow._sidecar_save_timer` (see its own docstring) --
    force a pending debounced sidecar-file write to run immediately."""
    if window._sidecar_save_timer.isActive():
        window._sidecar_save_timer.stop()
        window._run_debounced_sidecar_save()


def _make_1d_peak_spectrum(x0=5.0, fwhm=0.05, amplitude=1.0, n_pts=400):
    """A real, fittable 1D peak (unlike ``_make_1d_dataset``'s tiny
    8-point linear-ramp placeholder, which has nothing peak-shaped for
    a spectral fit to actually converge on)."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    data = gaussian(ppm, x0, fwhm, amplitude)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    return NMRSpectrum(data=data, axes=[axis], metadata=metadata)


def _fit_spectrum_then_curve(window):
    """Both steps of the analysis workflow, for the currently
    selected (pseudo-2D) spectrum -- step 1 (Fit Spectrum, the peak
    fit) then step 2 (Fit Curve, the relaxation/diffusion fit, a
    context-menu action per item C -- not a button)."""
    window._on_fit_spectrum_clicked()
    index = window._selected_index()
    window._on_fit_curve_pseudo2d_action(index)


def test_main_window_analysis_status_for_non_pseudo2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert "pseudo-2D" in window.analysis_status_label.text()


def test_main_window_analysis_status_for_pseudo2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=5, n_f2=8), vendor="bruker")
    assert "5 slices" in window.analysis_status_label.text()


def test_main_window_add_and_remove_peak_row(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.5)
    window._add_peak_row(shift=6.0, fwhm=0.2, eta=0.3)
    assert window.peak_table.rowCount() == 2

    window.peak_table.setCurrentCell(0, 0)
    window._on_remove_peak_clicked()
    assert window.peak_table.rowCount() == 1
    assert window.peak_table.item(0, 0).text() == "6"


def test_main_window_auto_pick_peaks_populates_table(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=0.0, fwhm=1.0, eta=0.5)  # a stale manual row -- should be replaced, not appended to
    window._on_auto_pick_peaks_clicked()
    assert window.peak_table.rowCount() == 1
    assert float(window.peak_table.item(0, 0).text()) == pytest.approx(5.0, abs=0.02)


def test_main_window_auto_pick_peaks_shows_markers_on_spectrum(qapp):
    # Explicit user request: "Auto peak pick should leave a visual
    # indication on the spectrum" -- previously only the peak table
    # was populated, nothing was drawn on the canvas itself.
    window = MainWindow()
    # Peak markers only draw while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._on_auto_pick_peaks_clicked()
    assert len(window.canvas._peak_marker_lines) == 1
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([5.0], abs=0.02)

    # A fresh rebuild (e.g. reselecting) clears the stale markers
    # rather than erroring on a Line2D no longer attached to any axes.
    window.canvas.show_spectrum(window.session.spectra[window._selected_index()])
    assert window.canvas._peak_marker_lines == []


def test_main_window_manually_added_peak_shows_a_marker(qapp):
    # Real user report: "manual peak picking also needs triangle
    # markers" -- auto-pick already drew its own marker list directly
    # from the detection result, but a manual add (the Add Peak
    # button, or click-to-pick via _on_peak_picked, both of which go
    # through _add_peak_row) never called set_peak_markers at all, so
    # nothing appeared on the canvas until Auto-Pick happened to be
    # used instead.
    window = MainWindow()
    # Peak markers only draw while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    assert window.canvas._peak_marker_lines == []

    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    assert len(window.canvas._peak_marker_lines) == 1
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([5.0])

    window._add_peak_row(shift=6.0, fwhm=0.05, eta=0.5)
    assert len(window.canvas._peak_marker_lines) == 2


def test_main_window_peak_picked_via_click_shows_a_marker(qapp):
    # Same gap as the manual-add test above, exercised through the
    # actual click-to-pick path (_on_pick_peak_toggled -> canvas click
    # -> _on_peak_picked) rather than calling _add_peak_row directly.
    window = MainWindow()
    # Peak markers only draw while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window.pick_peak_button.setChecked(True)
    window._on_peak_picked(5.0)
    assert len(window.canvas._peak_marker_lines) == 1


def test_main_window_removing_a_peak_removes_its_marker(qapp):
    window = MainWindow()
    # Peak markers only draw while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    window._add_peak_row(shift=6.0, fwhm=0.05, eta=0.5)
    assert len(window.canvas._peak_marker_lines) == 2

    window.peak_table.setCurrentCell(0, 0)
    window._on_remove_peak_clicked()
    assert len(window.canvas._peak_marker_lines) == 1
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([6.0])


def test_main_window_editing_a_peak_shift_moves_its_marker(qapp):
    # peak_table.itemChanged is wired to _refresh_peak_markers, same
    # as region_table's own itemChanged -> _sync_region_shading --
    # typing a new shift directly into the table (not just add/
    # remove/auto-pick) should move the marker too.
    window = MainWindow()
    # Peak markers only draw while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([5.0])

    window.peak_table.item(0, 0).setText("6.5")
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([6.5])


def test_main_window_markers_only_show_on_the_fitting_tab(qapp):
    # Reported directly: "the peak picking triangles/integrated regions
    # should only be visible while in the spectral fitting tab". Left
    # on the Spectrum view after moving on, they read as leftover state
    # -- the same reasoning that already hides the phase pivot marker.
    window = MainWindow()
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05, n_pts=200))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    window._add_region_row(4.5, 5.5)

    def markers():
        return len(window.canvas._peak_marker_lines)

    def shading():
        return sum(1 for p in window.canvas.ax.patches
                   if p.get_gid() == window.canvas._REGION_PATCH_GID)

    assert markers() == 1 and shading() == 1

    # Move to another tab -- both go.
    for index in range(window.side_tabs.count()):
        if index != window._fitting_tab_index:
            window.side_tabs.setCurrentIndex(index)
            break
    assert markers() == 0 and shading() == 0

    # And come back when it does.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    assert markers() == 1 and shading() == 1
    # The definitions themselves are untouched -- only the drawing.
    assert window.peak_table.rowCount() == 1
    assert window.region_table.rowCount() == 1


def test_main_window_align_echo_is_not_offered_on_the_indirect_dimension(qapp, tmp_path, monkeypatch):
    # Reported directly: "align echo should generally only apply to the
    # F2 dimension in a 2D". The reader already only applies it in its
    # own F2 block; offering the control on F1 implied otherwise.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=64), vendor="bruker")
    window.process_indirect_button.setChecked(True)      # a genuine 2D
    assert window._phase_mode == "2d"

    window.dim_combo.setCurrentIndex(0)                  # Direct (F2)
    assert window.align_echo_button.isHidden() is False

    window.dim_combo.setCurrentIndex(1)                  # Indirect (F1)
    assert window.align_echo_button.isHidden() is True

    window.dim_combo.setCurrentIndex(0)
    assert window.align_echo_button.isHidden() is False


def test_main_window_peak_markers_survive_reselection(qapp):
    # Real, related bug found while fixing the manual-peak-marker gap
    # above: MplCanvas.show_spectrum's own figure.clear() (run by
    # every reselection) discards the previous marker Line2D
    # references -- _restore_analysis_state's own peak-table rebuild
    # ran *before* that redraw (see _refresh_selection_display's own
    # call order), so its refresh was silently thrown away by the
    # very next line. Fixed by re-syncing again after
    # _draw_selected_spectrum, not just from _restore_analysis_state.
    window = MainWindow()
    # These markers only show while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05, n_pts=200))
    index_a = window._selected_index()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    assert len(window.canvas._peak_marker_lines) == 1

    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=3.0, fwhm=0.05, n_pts=200))
    index_b = window._selected_index()
    assert index_b != index_a
    assert window.canvas._peak_marker_lines == []  # B has no peaks of its own

    window._select_single_row(index_a)
    assert len(window.canvas._peak_marker_lines) == 1
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([5.0])


def test_main_window_region_shading_survives_reselection(qapp):
    # Same underlying bug as the peak-marker one above, same fix --
    # confirmed via direct reproduction before fixing (region shading
    # silently vanished on reselecting back to a spectrum that had a
    # saved region, even though the region_table itself still had the
    # row).
    window = MainWindow()
    # Region shading only shows while the Spectral Fitting tab is up.
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05, n_pts=200))
    index_a = window._selected_index()
    window._add_region_row(4.5, 5.5)

    def region_patch_count():
        return sum(1 for p in window.canvas.ax.patches if p.get_gid() == window.canvas._REGION_PATCH_GID)

    assert region_patch_count() == 1

    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=3.0, fwhm=0.05, n_pts=200))
    assert region_patch_count() == 0

    window._select_single_row(index_a)
    assert region_patch_count() == 1


def test_main_window_auto_pick_peaks_not_supported_for_conventional_2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt out of the pseudo-2D default

    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))
    window._on_auto_pick_peaks_clicked()
    assert shown.get("called") is True


def test_main_window_auto_pick_peaks_uses_prominence_spin(qapp):
    # A strong peak (4.0) plus a much weaker one (6.5, 8% the height) --
    # the default prominence (10%) should drop the weak one as noise,
    # while a lower one picks up both.
    ppm = np.linspace(3.0, 8.0, 2000)
    data = gaussian(ppm, 4.0, 0.05, 1.0) + gaussian(ppm, 6.5, 0.05, 0.08)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    spectrum = NMRSpectrum(data=data, axes=[axis], metadata=metadata)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    # In percent now, not a fraction of 1 -- see the box's own comment.
    assert window.auto_pick_prominence_spin.value() == pytest.approx(10.0)
    window._on_auto_pick_peaks_clicked()
    assert window.peak_table.rowCount() == 1  # weak peak dropped at the default

    window.auto_pick_prominence_spin.setValue(3.0)
    window._on_auto_pick_peaks_clicked()
    assert window.peak_table.rowCount() == 2  # both found at a lower threshold


def test_main_window_peak_pick_threshold_reaches_below_a_solvent_peak(qapp):
    # Reported directly: "minimum value for e.g. peak picking
    # threshold, 0.1% is not small enough with dominant solvent peaks".
    # Prominence is measured against the spectrum's own peak-to-peak
    # range, so a solvent peak a thousand times taller than the analyte
    # puts every real peak far below a tenth of a percent of it.
    ppm = np.linspace(0.0, 10.0, 8000)
    data = (gaussian(ppm, 4.7, 0.03, 1000.0)      # solvent
            + gaussian(ppm, 2.0, 0.03, 0.5)       # analyte, 0.05% of it
            + gaussian(ppm, 7.5, 0.03, 0.4))
    spectrum = NMRSpectrum(
        data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    # The box can actually express the number needed...
    assert window.auto_pick_prominence_spin.minimum() <= 0.001
    window.auto_pick_prominence_spin.setValue(0.01)
    assert window.auto_pick_prominence_spin.value() == pytest.approx(0.01)

    window._on_auto_pick_peaks_clicked()
    # ...and finds the analyte peaks the old 1% floor could not reach.
    found = sorted(float(window.peak_table.item(r, 0).text())
                   for r in range(window.peak_table.rowCount()))
    assert any(abs(shift - 2.0) < 0.1 for shift in found)
    assert any(abs(shift - 7.5) < 0.1 for shift in found)


def test_main_window_whittaker_is_offered_and_recorded(qapp, tmp_path, monkeypatch):
    # Explicitly requested, and flagged as the priority of its round.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256), vendor="bruker")
    index = window._selected_index()

    assert window.baseline_method_combo.findData("whittaker") >= 0
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("whittaker"))

    raw = window.session.spectra[index].metadata.raw
    assert raw["baseline_method"] == "whittaker"
    # Its own knob is the one shown; the others belong to other methods.
    assert window.baseline_lam_spin.isHidden() is False
    assert window.baseline_degree_spin.isHidden() is True
    assert window.baseline_smoothing_spin.isHidden() is True


def test_main_window_whittaker_lambda_is_auto_until_set(qapp, tmp_path, monkeypatch):
    # "Auto" has to reach baseline_correct as None, so it picks its own
    # point-count-scaled default rather than being pinned at 10**0.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256), vendor="bruker")
    index = window._selected_index()
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("whittaker"))

    assert window.baseline_lam_spin.value() == 0.0
    assert window.session.spectra[index].metadata.raw["baseline_lam"] is None

    window.baseline_lam_spin.setValue(9.0)
    assert window.session.spectra[index].metadata.raw["baseline_lam"] == pytest.approx(1e9)


def _pseudo2d_with_differing_baselines(n=1024, rows=6):
    """Rows whose baselines genuinely differ from one another -- the
    case a per-row fit must handle and a single shared fit cannot."""
    ppm = np.linspace(-2.0, 12.0, n)
    rng = np.random.default_rng(0)

    def lor(centre, width, amp):
        return amp / (1.0 + ((ppm - centre) / width) ** 2)

    peaks = lor(4.7, 0.02, 50.0) + lor(1.2, 0.01, 1.0) + lor(7.3, 0.01, 0.8)
    truths, data = [], []
    for r in range(rows):
        base = (0.1 + 0.05 * r) * np.sin(2 * np.pi * (ppm + 2.0) / (18.0 + r)) + 0.01 * r * ppm
        truths.append(base)
        data.append(peaks * np.exp(-0.15 * r) + base + rng.normal(0.0, 0.003, n))
    spectrum = NMRSpectrum(
        data=np.array(data),
        axes=[NMRAxis(kind="index", index=np.arange(1, rows + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    return ppm, spectrum, np.array(truths)


def test_baseline_across_a_pseudo2d_fits_every_row_on_its_own(qapp):
    # Asked directly: "check how the new baselining works across
    # pseudo-2D data". Each row gets its own fit, and a row fitted
    # inside the series must equal that row fitted alone.
    ppm, spectrum, truths = _pseudo2d_with_differing_baselines()
    spectrum.metadata.raw.update({"baseline_correct": True, "baseline_method": "whittaker"})

    corrected = MainWindow._baseline_correct_if_enabled_multi_row(
        spectrum.data, ppm, spectrum)
    alone = MainWindow._baseline_correct_if_enabled(spectrum.data[3], ppm, spectrum)
    np.testing.assert_allclose(corrected[3], alone)

    # And each row's own baseline is actually recovered, not one
    # compromise shared across rows that disagree.
    fitted = spectrum.data - corrected
    for row in range(spectrum.data.shape[0]):
        error = np.sqrt(np.mean((fitted[row] - truths[row]) ** 2))
        assert error < 0.05 * np.ptp(truths[row])


def test_whittaker_beats_the_others_across_a_pseudo2d(qapp):
    ppm, spectrum, truths = _pseudo2d_with_differing_baselines()

    def worst_row_error(method, **extra):
        spectrum.metadata.raw.update(
            {"baseline_correct": True, "baseline_method": method, **extra})
        corrected = MainWindow._baseline_correct_if_enabled_multi_row(
            spectrum.data, ppm, spectrum)
        fitted = spectrum.data - corrected
        return max(np.sqrt(np.mean((fitted[r] - truths[r]) ** 2))
                   for r in range(spectrum.data.shape[0]))

    whittaker = worst_row_error("whittaker")
    assert whittaker < worst_row_error("linear")
    assert whittaker < worst_row_error("polynomial", baseline_degree=3)


def test_main_window_baseline_overlay_matches_what_would_be_subtracted(qapp, tmp_path, monkeypatch):
    # Asked directly: "add a processing overlay to the plot showing the
    # baseline corrections before they are committed".
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")
    # Uncommitted is the whole point: with the correction switched off,
    # the baseline is still drawn so it can be judged first.
    window.baseline_correct_button.setChecked(False)
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("polynomial"))
    assert window.session.spectra[0].metadata.raw["baseline_correct"] is False

    # The Processing preview, not the Spectrum tab: an uncommitted
    # correction belongs in the speculative view, the same reasoning as
    # the ghost trace. Reported directly: "I still see the baseline
    # dashed blue line on the spectrum tab".
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    preview = window.processing_preview_canvas
    lines = [ln for ln in preview._pivot_marker_ax.lines
             if ln.get_gid() == preview._BASELINE_PREVIEW_GID]
    assert len(lines) == 1, "the baseline should be drawn without committing to it"
    # And never on the Spectrum tab.
    assert [ln for ln in window.canvas.ax.lines
            if ln.get_gid() == window.canvas._BASELINE_PREVIEW_GID] == []

    # It is the real thing, on the trace's own scale -- not a sketch.
    spectrum = window.session.spectra[0]
    ppm = spectrum.axis_values(0, "ppm")
    expected = processing_ft.compute_baseline(
        ppm, np.real(spectrum.data), **MainWindow._baseline_kwargs(spectrum))
    scale = float(np.max(np.abs(np.real(spectrum.data))))
    np.testing.assert_allclose(lines[0].get_ydata(), expected / scale)


def test_main_window_baseline_overlay_shows_what_was_removed(qapp, tmp_path, monkeypatch):
    # Reported directly: "seeing the baseline plotted against the
    # already corrected spectrum isn't so useful". Estimating a
    # baseline from an already-corrected trace produces a flat line at
    # zero; the overlay now shows the baseline that was *subtracted*,
    # over a faint reconstruction of the trace it was fitted to.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")
    # The ghost belongs to the *speculative* view, so that is where it
    # is drawn -- explicit user request, "the ghost spectrum should
    # only appear in the processing preview".
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    canvas = window.processing_preview_canvas

    def lines(gid, target=None):
        ax = (target or canvas)._pivot_marker_ax
        return [ln for ln in ax.lines if ln.get_gid() == gid]

    # Uncorrected: baseline drawn, no ghost needed (it would duplicate
    # the main trace).
    assert lines(canvas._BASELINE_PREVIEW_GID)
    assert not lines(canvas._BASELINE_GHOST_GID)

    window.baseline_correct_button.setChecked(True)

    baseline = np.array(lines(canvas._BASELINE_PREVIEW_GID)[0].get_ydata())
    ghost = lines(canvas._BASELINE_GHOST_GID)
    assert ghost, "the trace the baseline was fitted to should be shown"
    ghost_y = np.array(ghost[0].get_ydata())
    # Dashed, so it cannot be read as a second spectrum.
    assert ghost[0].get_linestyle() not in ("-", "solid")

    # The ghost is the trace the baseline was actually fitted to --
    # stored as it was used, not reconstructed as "displayed +
    # baseline", which assumed the displayed trace was the corrected
    # one and produced corrected + 2x baseline where it was not.
    corrected = np.array(canvas._pivot_marker_ax.lines[0].get_ydata())
    assert len(corrected) == len(ghost_y)
    assert not np.allclose(ghost_y, corrected)      # genuinely a different trace
    # Taking the baseline off the ghost lands on what is displayed.
    np.testing.assert_allclose(ghost_y - baseline, corrected, rtol=0.05, atol=1e-3)

    # And it is a real baseline, not the flat line a second pass gives.
    assert np.ptp(baseline) > 0

    # The Spectrum tab gets neither: it shows the spectrum as it *is*.
    assert [ln for ln in window.canvas.ax.lines
            if ln.get_gid() in (window.canvas._BASELINE_GHOST_GID,
                                window.canvas._BASELINE_PREVIEW_GID)] == []


def test_main_window_baseline_overlay_can_be_turned_off(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")

    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    preview = window.processing_preview_canvas

    def overlay_count():
        return len([ln for ln in preview._pivot_marker_ax.lines
                    if ln.get_gid() == preview._BASELINE_PREVIEW_GID])

    assert overlay_count() == 1
    window.baseline_preview_checkbox.setChecked(False)
    assert overlay_count() == 0
    window.baseline_preview_checkbox.setChecked(True)
    assert overlay_count() == 1


def test_main_window_baseline_overlay_follows_the_method(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")

    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    preview = window.processing_preview_canvas

    def overlay_y():
        line = next(ln for ln in preview._pivot_marker_ax.lines
                    if ln.get_gid() == preview._BASELINE_PREVIEW_GID)
        return np.array(line.get_ydata())

    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("linear"))
    linear = overlay_y()
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("whittaker"))
    assert not np.allclose(linear, overlay_y())


def test_main_window_sinusoidal_options_are_exposed_and_recorded(qapp, tmp_path, monkeypatch):
    # Asked directly: "add some controllable options to adjust the
    # polynomial and sinusoidal baselines".
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("sinusoidal"))
    raw = window.session.spectra[0].metadata.raw

    assert window.baseline_harmonics_spin.isHidden() is False
    assert window.baseline_period_spin.isHidden() is False
    assert window.baseline_degree_spin.isHidden() is True

    window.baseline_harmonics_spin.setValue(3)
    assert raw["baseline_harmonics"] == 3
    assert raw["baseline_period"] is None          # 0.0 means Auto
    window.baseline_period_spin.setValue(4.0)
    assert raw["baseline_period"] == pytest.approx(4.0)


def test_main_window_baseline_options_are_restored_per_spectrum(qapp, tmp_path, monkeypatch):
    # Lambda was never restored when it was added, so it carried the
    # previous spectrum's stiffness onto the next one.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512, name="a"), vendor="bruker")
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512, name="b"), vendor="bruker")

    window._select_single_row(0)
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("whittaker"))
    window.baseline_lam_spin.setValue(9.0)

    window._select_single_row(1)
    assert window.baseline_lam_spin.value() == 0.0          # b's own default
    window._select_single_row(0)
    assert window.baseline_lam_spin.value() == pytest.approx(9.0)


def test_main_window_warns_about_a_high_polynomial_degree(qapp, tmp_path, monkeypatch):
    # Asked directly: "add appropriate warnings for any of the more
    # automated steps that outline the potential issues or concerns".
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("polynomial"))

    window.baseline_degree_spin.setValue(2)
    assert "oscillates" not in window.processing_status_label.text()

    window.baseline_degree_spin.setValue(8)
    assert "oscillates" in window.processing_status_label.text()


def test_main_window_warns_when_a_peak_sits_in_the_edge_region(qapp):
    # Every method except Whittaker takes its baseline points from the
    # spectrum's own edges. Whether that is safe is a checkable
    # property of the data, not a standing caveat.
    ppm = np.linspace(0.0, 10.0, 2000)
    at_edge = gaussian(ppm, 9.8, 0.05, 1.0) + gaussian(ppm, 5.0, 0.05, 1.0)
    window = MainWindow()
    _load_spectrum_directly(window, NMRSpectrum(
        data=at_edge, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)))

    # Whittaker first: it does not use edge points, so it does not get
    # the warning -- and going via it guarantees the *next* setCurrentIndex
    # actually emits, since "linear" is the combo's own default.
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("whittaker"))
    assert "edge region" not in window.processing_status_label.text()

    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("linear"))
    assert "edge region" in window.processing_status_label.text()


def test_main_window_warns_when_auto_pick_is_asked_to_pick_noise(qapp):
    # The prominence floor was lowered to 0.001% on purpose; the same
    # setting will happily pick noise, and the table gives no hint.
    rng = np.random.default_rng(0)
    ppm = np.linspace(0.0, 10.0, 4000)
    data = gaussian(ppm, 5.0, 0.04, 1.0) + rng.normal(0.0, 0.01, ppm.size)
    window = MainWindow()
    _load_spectrum_directly(window, NMRSpectrum(
        data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)))

    window.auto_pick_prominence_spin.setValue(10.0)
    window._on_auto_pick_peaks_clicked()
    assert "likely noise" not in window.fitting_status_label.text()

    window.auto_pick_prominence_spin.setValue(0.05)     # well under the noise
    window._on_auto_pick_peaks_clicked()
    status = window.fitting_status_label.text()
    assert "likely noise" in status
    assert "σ" in status                                 # it says what the noise is


def test_main_window_processing_report_states_what_was_applied(qapp, tmp_path, monkeypatch):
    # Reported directly: "maybe a processing report for consistency
    # and/or for publication comparison".
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256), vendor="bruker")
    exponential = next(i for i in range(window.window_combo.count())
                       if window.window_combo.itemData(i) == "exponential")
    window.window_combo.setCurrentIndex(exponential)
    window.lb_spin.setValue(5.0)

    shown = {}
    monkeypatch.setattr(_ProcessingReportDialog, "exec",
                        lambda self: shown.setdefault("text", self.text_edit.toPlainText()))
    window._on_processing_report_clicked()

    assert "5 Hz of line broadening" in shown["text"]
    assert "exponential (EM)" in shown["text"]


def test_main_window_processing_report_compares_a_multi_selection(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256, name="one"), vendor="bruker")
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256, name="two"), vendor="bruker")
    exponential = next(i for i in range(window.window_combo.count())
                       if window.window_combo.itemData(i) == "exponential")
    window.window_combo.setCurrentIndex(exponential)
    window.lb_spin.setValue(3.0)          # applies to the selected one only
    _select_rows(window, [0, 1])

    shown = {}
    monkeypatch.setattr(_ProcessingReportDialog, "exec",
                        lambda self: shown.setdefault("text", self.text_edit.toPlainText()))
    window._on_processing_report_clicked()

    # The "for consistency" half: the setting they disagree on is marked.
    assert "differs" in shown["text"]


def test_main_window_clear_all_peaks_button(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.5)
    window._add_peak_row(shift=6.0, fwhm=0.2, eta=0.3)
    assert window.peak_table.rowCount() == 2
    window._on_clear_peaks_clicked()  # what the "Clear All" button/menu action actually call
    assert window.peak_table.rowCount() == 0


def test_main_window_remove_peak_with_no_selection_is_noop(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.5)
    window.peak_table.setCurrentCell(-1, -1)
    window._on_remove_peak_clicked()
    assert window.peak_table.rowCount() == 1


def test_main_window_pick_peak_button_toggles_canvas_pick_mode(qapp):
    window = MainWindow()
    assert window.canvas._pick_mode is False
    window.pick_peak_button.setChecked(True)
    assert window.canvas._pick_mode is True
    window.pick_peak_button.setChecked(False)
    assert window.canvas._pick_mode is False


def test_main_window_pick_peak_button_switches_to_spectrum_tab(qapp, tmp_path):
    # "Pick from Plot" doesn't work if the Spectrum tab (the only one
    # pick mode actually listens on) isn't the one visible -- e.g.
    # left showing Compare/Fit/Processing-preview from an earlier
    # interaction, all reachable without leaving the Analysis tab this
    # button lives on.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)

    window.pick_peak_button.setChecked(True)

    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_peak_picked_adds_row_only_when_pick_mode_active(qapp):
    window = MainWindow()
    window.canvas.peakPicked.emit(5.5)  # pick mode off -- should be ignored
    assert window.peak_table.rowCount() == 0


def test_main_window_select_region_button_switches_to_spectrum_tab(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)

    window.select_region_button.setChecked(True)

    assert window.canvas_tabs.currentWidget() is window.canvas
    assert window.canvas._region_select_mode is True


def test_main_window_region_selected_adds_a_row(qapp):
    window = MainWindow()
    window.canvas.regionSelected.emit(3.0, 7.0)
    assert window.region_table.rowCount() == 1
    assert window._read_region_defs() == [(3.0, 7.0)]


def test_main_window_select_region_menu_action_syncs_with_button(qapp):
    window = MainWindow()
    assert window.select_region_menu_action.shortcut().toString() in ("Ctrl+I", "⌘I")

    window.select_region_button.setChecked(True)
    assert window.select_region_menu_action.isChecked() is True

    window.select_region_menu_action.setChecked(False)
    assert window.select_region_button.isChecked() is False


def test_main_window_measurement_menu_action_is_cmd_d_and_reaches_every_canvas(qapp):
    # Two real reports: the shortcut had become a bare "d" (a
    # matplotlib key binding on the canvas) rather than the Cmd+D that
    # was asked for, and measurement mode only ever reached the
    # Spectrum canvas -- so on any other tab it did nothing.
    window = MainWindow()
    assert window.measurement_menu_action.shortcut().toString() in ("Ctrl+D", "⌘D")

    window.measurement_menu_action.setChecked(True)
    canvases = window._measurement_canvases()
    assert len(canvases) > 1
    assert all(c._measurement_mode for c in canvases)

    window.measurement_menu_action.setChecked(False)
    assert not any(c._measurement_mode for c in canvases)

    # A canvas's own toolbar button drives the menu item, which is what
    # then applies it everywhere -- so the two can never disagree.
    window.canvas._measure_button.setChecked(True)
    assert window.measurement_menu_action.isChecked() is True
    assert all(c._measurement_mode for c in canvases)


def test_main_window_crosshair_toggle_reaches_every_canvas(qapp):
    # "The crosshair button disappeared in the last round."
    window = MainWindow()
    assert window.crosshair_menu_action.isChecked() is True
    canvases = window._measurement_canvases()
    assert all(c._crosshair_enabled for c in canvases)

    window.crosshair_menu_action.setChecked(False)
    assert not any(c._crosshair_enabled for c in canvases)
    assert window.canvas._crosshair_button.isChecked() is False

    window.canvas._crosshair_button.setChecked(True)
    assert window.crosshair_menu_action.isChecked() is True
    assert all(c._crosshair_enabled for c in canvases)


def test_main_window_measurement_result_goes_to_the_status_bar(qapp):
    # It used to print() to stdout, which is why "measurement isn't
    # coming up" -- nothing reached the GUI at all.
    window = MainWindow()
    window._on_measurement_completed({"summary": "\u0394x = 0.0185 ppm / 7.42 Hz"})
    assert "7.42 Hz" in window.statusBar().currentMessage()


def test_main_window_region_table_add_remove_clear(qapp):
    window = MainWindow()
    window._add_region_row(1.0, 2.0)
    window._add_region_row(3.0, 4.0)
    assert window._read_region_defs() == [(1.0, 2.0), (3.0, 4.0)]

    window.region_table.setCurrentCell(0, 0)
    window._on_remove_region_clicked()
    assert window._read_region_defs() == [(3.0, 4.0)]

    window._on_clear_regions_clicked()
    assert window.region_table.rowCount() == 0


def test_main_window_integrate_regions_feeds_fit_curve_recovers_known_t2(qapp):
    # The end-to-end promise: no peak fit needed at all -- a region
    # integral per slice, straight into the same Fit Curve pathway a
    # peak fit would use.
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 10)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_region_row(4.5, 5.5)
    window._on_integrate_regions_clicked()

    assert "Integrated" in window.integration_results_label.text()
    assert window.view_slice_fit_spin.isEnabled() is False  # no peak-fit-shaped viewer for this
    # "region", not the old ambiguous "integral": these results are a
    # raw sum over a region with no lineshape fitted to it.
    assert window.value_source_combo.currentData() == "region"

    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"
    window._on_fit_curve_button_clicked()

    result = window._relaxation_fit_results[0]
    assert result["params"]["T2"] == pytest.approx(true_t2, rel=0.05)


def test_main_window_integrate_regions_requires_at_least_one_region(qapp, tmp_path, monkeypatch):
    shown = {}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._on_integrate_regions_clicked()  # must not raise, must not pop a real dialog
    assert shown.get("called") is True
    assert window._slice_fit_results is None


def test_main_window_regions_bound_the_peak_fit(qapp, monkeypatch):
    from opennmr.gui import main_window as window_module
    from opennmr.processing import fitting as processing_fitting

    ppm = np.linspace(0.0, 10.0, 400)
    data = gaussian(ppm, 5.0, 0.1, 1.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    spectrum = NMRSpectrum(data=data, axes=[axis], metadata=metadata)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.0)
    window._add_region_row(4.0, 6.0)

    captured = {}
    real_fit_slices = processing_fitting.NMR2D_fit_slices

    def spy(*args, **kwargs):
        captured["fit_regions"] = kwargs.get("fit_regions")
        return real_fit_slices(*args, **kwargs)

    monkeypatch.setattr(window_module.processing_fitting, "NMR2D_fit_slices", spy)
    window._fit_peaks([spectrum])

    assert captured["fit_regions"] == [(4.0, 6.0)]


def test_main_window_restoring_integration_results_does_not_crash_slice_fit_viewer(qapp):
    # _display_slice_fit needs a real peak-fit shape (popt/pcov/
    # peak_defs/y_fit) -- an Integrate Regions result stored in the
    # exact same metadata.raw["slice_fit_results"] slot must not reach
    # it when a spectrum carrying one is reselected.
    delays = np.geomspace(0.001, 0.3, 6)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_region_row(4.5, 5.5)
    window._on_integrate_regions_clicked()

    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)  # a second, unrelated spectrum
    other = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.08)
    _load_spectrum_directly(window, other)

    window._select_single_row(0)  # back to the integration-results spectrum -- must not raise
    assert window.view_slice_fit_spin.isEnabled() is False


def test_main_window_read_peak_defs_parses_table(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)
    peak_defs = window._read_peak_defs()
    assert len(peak_defs) == 1
    peak_def = peak_defs[0]
    assert {k: peak_def[k] for k in ("type", "shift", "fwhm", "eta")} == {
        "type": "pseudo_voigt", "shift": 5.1, "fwhm": 0.2, "eta": 0.5,
    }
    # Default position tolerance is 0.5x FWHM (see item 8: unconstrained
    # position bounds let a fit slide onto a neighbouring peak).
    lower, upper = peak_def["bounds"]["shift"]
    assert lower == pytest.approx(5.1 - 0.1)
    assert upper == pytest.approx(5.1 + 0.1)


def test_main_window_read_peak_defs_raises_on_bad_value(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)
    window.peak_table.item(0, 1).setText("not a number")
    with pytest.raises(ValueError):
        window._read_peak_defs()


def test_main_window_position_tolerance_spin_widens_bounds(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)
    window.position_tolerance_spin.setValue(2.0)
    lower, upper = window._read_peak_defs()[0]["bounds"]["shift"]
    assert lower == pytest.approx(5.1 - 0.4)
    assert upper == pytest.approx(5.1 + 0.4)


def test_main_window_peak_table_fixed_checkbox_locks_every_parameter(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)
    window.peak_table.item(0, 3).setCheckState(Qt.CheckState.Checked)

    peak_def = window._read_peak_defs()[0]

    assert peak_def["fixed"] == {"shift": True, "fwhm": True, "A": True, "eta": True}


def test_main_window_peak_table_fixed_checkbox_unchecked_by_default(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)

    peak_def = window._read_peak_defs()[0]

    assert "fixed" not in peak_def


def test_main_window_peak_table_per_parameter_fixed(qapp):
    # Real user requirement: "individual parameters (not just an entire
    # line) should be fixable." The row's own all-or-nothing "Fixed"
    # checkbox (tested above) already covers "fix everything" -- this
    # is the finer-grained path, via _PEAK_FIXED_ITEM_ROLE (set by the
    # Set Bounds / Fixed Parameters… dialog; here direct, matching the
    # existing bounds test's own style).
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5, per_param_fixed={"shift": True})

    peak_def = window._read_peak_defs()[0]

    assert peak_def["fixed"] == {"shift": True}


def test_main_window_peak_table_row_fixed_checkbox_overrides_per_parameter_fixed(qapp):
    # _on_set_peak_bounds's own documented precedence: the row's
    # all-or-nothing checkbox wins outright over any per-parameter
    # choice, even a conflicting one (here: FWHM explicitly left free
    # per-parameter, but the row checkbox still fixes it too).
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5, per_param_fixed={"shift": True})
    window.peak_table.item(0, 3).setCheckState(Qt.CheckState.Checked)

    peak_def = window._read_peak_defs()[0]

    assert peak_def["fixed"] == {"shift": True, "fwhm": True, "A": True, "eta": True}


def test_main_window_set_peak_bounds_dialog_stores_per_parameter_fixed(qapp, monkeypatch):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)

    def fake_exec(dialog):
        checks = dialog.findChildren(QCheckBox)
        checks[0].setChecked(True)  # shift, per _PEAK_BOUNDS_FIELDS' own order
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    window._on_set_peak_bounds(0)

    stored = window.peak_table.item(0, 0).data(window._PEAK_FIXED_ITEM_ROLE)
    assert stored == {"shift": True}
    peak_def = window._read_peak_defs()[0]
    assert peak_def["fixed"] == {"shift": True}


def test_main_window_set_peak_bounds_dialog_prefills_existing_per_parameter_fixed(qapp, monkeypatch):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5, per_param_fixed={"fwhm": True})

    def fake_exec(dialog):
        checks = dialog.findChildren(QCheckBox)
        assert checks[1].isChecked()  # fwhm, per _PEAK_BOUNDS_FIELDS' own order
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    window._on_set_peak_bounds(0)


def test_main_window_fit_updating_peak_table_preserves_per_parameter_fixed(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5, per_param_fixed={"shift": True})
    result = {
        "popt": np.array([5.02, 0.05, 1.0, 0.5]),
        "pcov": np.zeros((4, 4)),
        "peak_defs": window._read_peak_defs(),
        "is_free": np.array([True, True, True, True]),
    }

    window._update_peak_table_from_fit(result)

    stored = window.peak_table.item(0, 0).data(window._PEAK_FIXED_ITEM_ROLE)
    assert stored == {"shift": True}


def test_main_window_peak_custom_bounds_override_default_shift_bound(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)
    window.peak_table.item(0, 0).setData(Qt.ItemDataRole.UserRole, {"shift": (4.0, 6.0), "eta": (0.1, 0.9)})

    peak_def = window._read_peak_defs()[0]

    assert peak_def["bounds"]["shift"] == (4.0, 6.0)
    assert peak_def["bounds"]["eta"] == (0.1, 0.9)


def test_main_window_set_peak_bounds_dialog_stores_parsed_bounds(qapp, monkeypatch):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)

    def fake_exec(dialog):
        edits = dialog.findChildren(QLineEdit)
        edits[0].setText("4.5, 5.5")  # shift
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    window._on_set_peak_bounds(0)

    stored = window.peak_table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert stored == {"shift": (4.5, 5.5)}


def test_main_window_set_peak_bounds_dialog_cancelled_leaves_bounds_unchanged(qapp, monkeypatch):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5, bounds={"shift": (4.0, 6.0)})

    monkeypatch.setattr(QDialog, "exec", lambda dialog: QDialog.DialogCode.Rejected)
    window._on_set_peak_bounds(0)

    stored = window.peak_table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert stored == {"shift": (4.0, 6.0)}


def test_main_window_fit_updating_peak_table_preserves_custom_bounds(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5, bounds={"shift": (4.5, 5.5)})
    result = {
        "popt": np.array([5.02, 0.05, 1.0, 0.5]),
        "pcov": np.zeros((4, 4)),
        "peak_defs": window._read_peak_defs(),
        "is_free": np.array([True, True, True, True]),
    }

    window._update_peak_table_from_fit(result)

    stored = window.peak_table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert stored == {"shift": (4.5, 5.5)}


def test_main_window_sn_gating_kwargs_empty_when_no_noise_region(qapp):
    window = MainWindow()
    assert window._read_sn_gating_kwargs() == {}


def test_main_window_sn_gating_kwargs_parses_region_and_threshold(qapp):
    window = MainWindow()
    window.sn_noise_region_edit.setText("8.5, 9.5")
    window.sn_threshold_spin.setValue(5.0)
    assert window._read_sn_gating_kwargs() == {"noise_region": (8.5, 9.5), "sn_threshold": 5.0}


def test_main_window_sn_gating_kwargs_raises_on_bad_region(qapp):
    window = MainWindow()
    window.sn_noise_region_edit.setText("not a range")
    with pytest.raises(ValueError):
        window._read_sn_gating_kwargs()


def test_main_window_analysis_state_round_trips_fixed_and_bounds(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._add_peak_row(
        shift=5.0, fwhm=0.1, eta=0.5, fixed=True, bounds={"shift": (4.5, 5.5)}, per_param_fixed={"eta": True},
    )
    window.sn_noise_region_edit.setText("8, 9")
    window.sn_threshold_spin.setValue(4.0)
    window._save_current_analysis_state()

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")  # switch away
    window._select_single_row(0)  # switch back -- triggers _restore_analysis_state

    assert window.peak_table.item(0, 3).checkState() == Qt.CheckState.Checked
    assert window.peak_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == {"shift": (4.5, 5.5)}
    assert window.peak_table.item(0, 0).data(window._PEAK_FIXED_ITEM_ROLE) == {"eta": True}
    assert window.sn_noise_region_edit.text() == "8, 9"
    assert window.sn_threshold_spin.value() == pytest.approx(4.0)


def test_main_window_position_tolerance_zero_locks_but_does_not_crash(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.1, fwhm=0.2, eta=0.5)
    window.position_tolerance_spin.setValue(0.0)
    lower, upper = window._read_peak_defs()[0]["bounds"]["shift"]
    assert lower < 5.1 < upper  # strict, per scipy's bounds requirement
    assert upper - lower < 1e-6


def test_main_window_position_tolerance_siloed_per_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window.position_tolerance_spin.setValue(3.0)

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    assert window.position_tolerance_spin.value() == pytest.approx(0.5)  # fresh spectrum -> default

    window._select_single_row(0)
    assert window.position_tolerance_spin.value() == pytest.approx(3.0)


def test_main_window_resolve_delays_manual_list(qapp, tmp_path):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText("1, 2, 3")
    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [1.0, 2.0, 3.0])


def test_main_window_manual_list_is_read_in_the_selected_unit(qapp):
    # Typed values are in whatever the Unit combo says; the fit is
    # always in seconds. Milliseconds is the default, by request.
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText("10, 100, 1000")

    assert window.delay_unit_combo.currentText() == "ms"
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.01, 0.1, 1.0])

    _set_delay_unit(window, "s")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [10.0, 100.0, 1000.0])

    _set_delay_unit(window, "µs")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [1e-5, 1e-4, 1e-3])


def test_main_window_manual_list_suffix_beats_the_unit_combo(qapp):
    # "10m" is 10 ms whatever the combo says -- the vdlist convention.
    # Rescaling an explicit suffix by the combo as well would be the
    # one genuinely ambiguous reading.
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "µs")
    window.manual_delays_edit.setText("10m, 100m, 1s")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.01, 0.1, 1.0])


def test_main_window_delay_unit_is_hidden_for_a_diffusion_fit(qapp):
    # A b-value has no SI-prefix convention to pick from.
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)
    assert window.delay_unit_combo.isHidden()
    assert window.delay_unit_row_label.isHidden()
    window.fit_type_combo.setCurrentIndex(0)
    assert not window.delay_unit_combo.isHidden()


def test_main_window_resolve_delays_auto_falls_back_to_slice_index(qapp, tmp_path):
    # No vdlist available and Values left on Auto -- fits against plain
    # slice index (1, 2, 3, ...) rather than raising, per explicit request.
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1, 0.5, 1.0])
    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_main_window_resolve_delays_auto_reads_real_vdlist_file(qapp, tmp_path):
    (tmp_path / "vdlist").write_text("1m\n10m\n100m\n")
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(tmp_path),
    ))
    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays["seconds"], [0.001, 0.01, 0.1])


def test_main_window_resolve_delays_start_step(qapp, tmp_path):
    # 3 delays -> 3 slices; Start+step must generate exactly one value
    # per slice, not a separately-specified End that could mismatch.
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(2)  # Start + step
    _set_delay_unit(window, "s")
    window.delay_start_spin.setValue(0.0)
    window.delay_step_spin.setValue(0.25)
    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [0.0, 0.25, 0.5])


def test_main_window_start_step_is_read_in_the_selected_unit(qapp):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(2)
    _set_delay_unit(window, "ms")
    window.delay_start_spin.setValue(10.0)
    window.delay_step_spin.setValue(10.0)
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.01, 0.02, 0.03])
    # The spin boxes name the unit they are in.
    assert window.delay_start_spin.suffix().strip() == "ms"


def test_main_window_resolve_delays_vclist_times_factor(qapp, tmp_path):
    # Nothing picked in the panel: the source finds the experiment's
    # own list file, exactly as the Bruker-only vclist source it
    # generalises always did.
    expt = tmp_path / "expt"
    expt.mkdir()
    (expt / "vclist").write_text("1\n2\n4\n8\n")

    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt),
    ))
    window.delay_source_combo.setCurrentIndex(3)  # List file x factor
    # Bare counters, so list x factor is in the panel's Unit; seconds here.
    _set_delay_unit(window, "s")
    window.list_factor_edit.setText("0.5")

    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [0.5, 1.0, 2.0, 4.0])


def test_main_window_resolve_delays_list_file_any_vendor_filename(qapp, tmp_path):
    # The point of the generalisation: an X-Pulse-style valueslist in
    # the vendor's own parameters/ folder, read by the same source that
    # reads a Bruker vclist, with no vendor check anywhere.
    expt = tmp_path / "4"
    (expt / "parameters").mkdir(parents=True)
    (expt / "parameters" / "G1_valueslist.txt").write_text("0.1\n0.5\n0.9\n")

    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, vendor="xpulse", source_path=str(expt),
    ))
    window.delay_source_combo.setCurrentIndex(3)
    _set_delay_unit(window, "s")
    window.list_factor_edit.setText("2")

    values = window._resolve_independent_values(spectrum, is_diffusion=True)
    np.testing.assert_allclose(values, [0.2, 1.0, 1.8])


def test_main_window_resolve_delays_list_file_honours_time_suffixes(qapp, tmp_path):
    expt = tmp_path / "expt"
    expt.mkdir()
    (expt / "vdlist").write_text("# a comment\n10m\n\n100m\n1s\n")

    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt),
    ))
    window.delay_source_combo.setCurrentIndex(3)

    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [0.01, 0.1, 1.0])


def test_main_window_resolve_delays_list_file_none_found_explains_itself(qapp, tmp_path):
    expt = tmp_path / "empty"
    expt.mkdir()

    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt),
    ))
    window.delay_source_combo.setCurrentIndex(3)

    with pytest.raises(ValueError, match="No list file found"):
        window._resolve_independent_values(spectrum, is_diffusion=False)


def test_main_window_manual_list_accepts_whitespace_separated_values(qapp):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    window.manual_delays_edit.setText("10m 100m 1s")

    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [0.01, 0.1, 1.0])


def test_main_window_manual_list_row_label_hides_with_its_field(qapp):
    # The orphaned "List:" label was the visible half of the report
    # that the manual list "doesn't seem to be used": the field it
    # belongs to was hidden under every other source while the label
    # stayed put.
    window = MainWindow()
    window.delay_source_combo.setCurrentIndex(0)  # Auto
    assert window.manual_delays_row_label.isHidden()
    assert window.manual_delays_edit.isHidden()

    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    assert not window.manual_delays_row_label.isHidden()
    assert not window.manual_delays_edit.isHidden()


def test_main_window_resolve_delays_relative_gradient(qapp):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, raw={"gradient_relative": np.array([0.1, 0.5, 0.9])},
    ))
    window.delay_source_combo.setCurrentIndex(4)  # Relative gradient
    values = window._resolve_independent_values(spectrum, is_diffusion=True)
    np.testing.assert_allclose(values, [0.1, 0.5, 0.9])


def test_main_window_resolve_delays_relative_gradient_missing_raises(qapp):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    window.delay_source_combo.setCurrentIndex(4)  # Relative gradient
    with pytest.raises(ValueError):
        window._resolve_independent_values(spectrum, is_diffusion=True)


def test_main_window_fit_spectrum_works_for_1d(qapp):
    # Item A from the user: spectral fitting must be possible for a
    # plain 1D spectrum directly, not just as a side effect of fitting
    # a pseudo-2D series.
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)

    window._on_fit_spectrum_clicked()

    assert window.view_slice_fit_spin.isEnabled() is True
    assert window.view_slice_fit_spin.maximum() == 1
    assert window.canvas_tabs.currentWidget() is window.spectral_fit_canvas


def test_main_window_build_fit_report_csv_spectral_fit_only(qapp):
    # Fitting-wizard roadmap item: "a more readable, exportable fit
    # report." Spectral-fit-only case (no Fit Curve run yet) -- the
    # CSV should have exactly the spectral-fit section, correctly
    # parseable, with no "Curve fit" section at all.
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05, amplitude=1.0))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    window._on_fit_spectrum_clicked()

    spectrum = window.session.spectra[window._selected_index()]
    report = window._build_fit_report_csv(spectrum.metadata.raw["slice_fit_results"])

    rows = list(csv.reader(io.StringIO(report)))
    assert rows[0] == ["Spectral fit: row per slice per peak"]   # (reworded 2026-09-30)
    assert rows[1] == ["Slice", "Peak", "Type", "Shift", "Shift err", "FWHM", "FWHM err",
                        "Amplitude", "Amplitude err", "Eta", "Eta err"]
    assert rows[2][0:3] == ["1", "1", "pseudo_voigt"]
    assert float(rows[2][3]) == pytest.approx(5.0, abs=1e-2)  # shift
    assert float(rows[2][7]) == pytest.approx(1.0, abs=1e-1)  # amplitude
    assert "Curve fit" not in report


def test_main_window_build_fit_report_csv_includes_curve_fit_section(qapp):
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 12)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"

    _fit_spectrum_then_curve(window)

    report = window._build_fit_report_csv(
        spectrum.metadata.raw["slice_fit_results"], spectrum.metadata.raw.get("relaxation_fit_results"),
    )
    rows = list(csv.reader(io.StringIO(report)))
    assert sum(1 for r in rows[2:] if len(r) > 2 and r[2] == "pseudo_voigt") == len(delays)  # one row per slice
    curve_header_idx = rows.index(["Curve fit: row per peak"])   # (reworded 2026-09-30)
    # A stated uncertainty kind precedes the table: a column of "err"
    # numbers whose meaning depends on whether per-point errors were
    # supplied is what gets misread once the CSV leaves this program.
    assert rows[curve_header_idx + 1][0].startswith("Uncertainty:")
    header = rows[curve_header_idx + 2]
    assert header[0:2] == ["Peak", "Model"]
    assert "T2" in header
    assert "R^2" in header and "reduced chi^2" in header
    data_row = rows[curve_header_idx + 3]
    t2_value = float(data_row[header.index("T2")])
    assert t2_value == pytest.approx(true_t2, rel=0.1)


def test_main_window_export_fit_report_writes_csv_file(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    window._on_fit_spectrum_clicked()

    out_path = tmp_path / "report.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), "CSV files (*.csv)"))
    window._on_export_fit_report_clicked()

    assert out_path.exists()
    content = out_path.read_text()
    assert "Spectral fit" in content
    assert "pseudo_voigt" in content


def test_main_window_export_fit_report_prompts_when_nothing_fitted_yet(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))

    dialog_called = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: dialog_called.append(True))
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_export_fit_report_clicked()

    assert shown.get("called") is True
    assert dialog_called == []  # never even got to the save dialog


def _make_drifting_peak_series(n_slices=6, drift_per_slice=0.15, x0=5.0, fwhm=0.05, A=1.0, n_pts=800):
    """test_fitting.py's own helper of the same name, duplicated here
    (a plain NMRSpectrum builder, no GUI/Qt involved) rather than
    imported across test files -- a pseudo-2D series where the peak's
    true position drifts slice to slice, exercising seed_mode."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    true_shifts = [x0 + drift_per_slice * i for i in range(n_slices)]
    data = np.array([gaussian(ppm, s, fwhm, A) for s in true_shifts])
    f1_axis = NMRAxis(kind="index", index=np.arange(1, n_slices + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata), true_shifts


def test_main_window_seed_mode_defaults_to_fixing_the_lineshape(qapp):
    # Changed deliberately, on explicit user instruction: "for pseudo-2D
    # data ... the position and width should generally stay fixed and
    # let intensity vary across slices". That is what a relaxation or
    # diffusion series *is*, so it is the default rather than an option
    # buried in a collapsed Advanced block, which is where it used to
    # live and why it went unnoticed.
    window = MainWindow()
    assert window.seed_mode_combo.currentData() == "apply_first"
    # Every other mode is still there, unchanged.
    modes = {window.seed_mode_combo.itemData(i) for i in range(window.seed_mode_combo.count())}
    assert modes == {"apply_first", "independent", "propagate"}


def test_main_window_anchor_slice_controls_follow_the_seed_mode(qapp):
    window = MainWindow()
    assert window.anchor_slice_spin.isHidden() is False   # apply_first is the default
    window.seed_mode_combo.setCurrentIndex(
        next(i for i in range(window.seed_mode_combo.count())
             if window.seed_mode_combo.itemData(i) == "independent"))
    # Nothing is anchored to anything, so there is no anchor to pick.
    assert window.anchor_slice_spin.isHidden() is True


def test_main_window_series_group_only_shown_for_a_pseudo_2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.series_fit_group.isHidden() is True
    window._load(_make_2d_dataset(tmp_path, n_f1=6, n_f2=8), vendor="bruker")
    assert window.series_fit_group.isHidden() is False
    assert window.anchor_slice_spin.maximum() == 6


def test_main_window_anchor_first_and_last_buttons(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=7, n_f2=8), vendor="bruker")
    window.anchor_last_button.click()
    assert window.anchor_slice_spin.value() == 7
    window.anchor_first_button.click()
    assert window.anchor_slice_spin.value() == 1


def test_main_window_fit_spectrum_seed_mode_propagate_tracks_drift(qapp):
    # Fitting-wizard roadmap item: "fit one slice first, then... use
    # them as each next slice's own starting guess... rather than
    # every slice fit independently from scratch, today's behaviour."
    # End-to-end through the real GUI call path (_on_fit_spectrum_clicked
    # -> _fit_peaks -> NMR2D_fit_slices), not just the backend function
    # directly (already covered in test_fitting.py) -- confirms the
    # combo's own selection actually reaches the fit.
    #
    # position_tolerance_spin widened: unlike test_fitting.py's own
    # bare-peak_defs backend test (no bounds at all), the GUI always
    # attaches a shift bound from this control (_read_peak_defs), which
    # _reseed_peak_position recentres around each new seed but does not
    # widen -- left at its default (0.5xFWHM = 0.025 ppm half-width),
    # this series' own 0.15 ppm/slice drift is simply too large for a
    # handful of slices to "catch up" one narrow recentred window at a
    # time (confirmed directly: capped at ~5.125 ppm instead of the
    # true 5.75 with the default tolerance) -- a real, separate control
    # doing its own job correctly, not a seed_mode bug, and a genuinely
    # large drift like this needs that control widened either way, in
    # the GUI as much as in this test.
    spectrum, true_shifts = _make_drifting_peak_series()
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=true_shifts[0], fwhm=0.05, eta=0.0)
    window.position_tolerance_spin.setValue(20.0)
    window.seed_mode_combo.setCurrentIndex(window.seed_mode_combo.findData("propagate"))

    window._on_fit_spectrum_clicked()

    last_result = spectrum.metadata.raw["slice_fit_results"][-1]
    assert last_result["popt"][0] == pytest.approx(true_shifts[-1], abs=1e-2)


def test_main_window_seed_mode_round_trips_through_reselection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    # Deliberately not the default, so the round trip actually proves
    # something rather than agreeing with it by coincidence.
    window.seed_mode_combo.setCurrentIndex(window.seed_mode_combo.findData("propagate"))
    window._save_current_analysis_state()

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")  # switch away
    # Spectrum b has no saved state, so it gets the default.
    assert window.seed_mode_combo.currentData() == "apply_first"

    window._select_single_row(0)  # switch back -- triggers _restore_analysis_state
    assert window.seed_mode_combo.currentData() == "propagate"


def test_main_window_amplitude_bound_regions_checkbox_defaults_to_unchecked(qapp):
    window = MainWindow()
    assert not window.amplitude_bound_regions_checkbox.isChecked()


def test_main_window_fit_spectrum_amplitude_bound_regions_caps_the_fit(qapp):
    # Fitting-wizard roadmap item: "bounding possibly from integrals."
    # End-to-end through the real GUI call path (_on_fit_spectrum_clicked
    # -> _fit_peaks -> NMR2D_fit_slices), not just the backend function
    # directly (already covered in test_fitting.py). Region (4.95, 5.05)
    # is deliberately narrower than the true peak's own support
    # (fwhm=0.2) -- captures only a fraction of its true area, so
    # checking the box should visibly cap the fitted amplitude below
    # what the exact same fit finds unconstrained.
    x0, fwhm, true_A = 5.0, 0.2, 2.0
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=x0, fwhm=fwhm, amplitude=true_A))
    window._add_peak_row(shift=x0, fwhm=fwhm, eta=0.0)
    window._add_region_row(4.95, 5.05)

    window._on_fit_spectrum_clicked()
    unconstrained_A = window.session.spectra[0].metadata.raw["slice_fit_results"][0]["popt"][2]
    assert unconstrained_A == pytest.approx(true_A, rel=1e-2)

    window.amplitude_bound_regions_checkbox.setChecked(True)
    window._on_fit_spectrum_clicked()
    bounded_A = window.session.spectra[0].metadata.raw["slice_fit_results"][0]["popt"][2]
    assert bounded_A < unconstrained_A
    assert bounded_A < true_A


def test_main_window_amplitude_bound_regions_round_trips_through_reselection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window.amplitude_bound_regions_checkbox.setChecked(True)
    window._save_current_analysis_state()

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")  # switch away
    assert not window.amplitude_bound_regions_checkbox.isChecked()  # spectrum b has no saved state

    window._select_single_row(0)  # switch back -- triggers _restore_analysis_state
    assert window.amplitude_bound_regions_checkbox.isChecked()


# ============================================================
# Spectral-fit results table -- ported from upstream (Leo, 14a6fd3)
# ============================================================

def test_spectral_fit_report_is_a_table_matching_the_peak_table(qapp):
    # "a sensible table, basically matching the results of the peak pick
    # table but with the outputs" -- it was a line of text per peak.
    window = MainWindow()
    x = np.linspace(10.0, 0.0, 800)
    y = (processing_fitting.lorentzian(x, 3.0, 0.05, 1.0)
         + processing_fitting.lorentzian(x, 6.0, 0.05, 3.0))
    spectrum = NMRSpectrum(
        data=y, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0),
    )
    window._add_peak_row(3.0, 0.05)
    window._add_peak_row(6.0, 0.05)
    results = window._fit_peaks([spectrum])
    window._display_slice_fit(results[0])

    table = window.spectral_fit_table
    columns = [table.horizontalHeaderItem(c).text().replace("\n", " ")
               for c in range(table.columnCount())]
    # The peak table's own columns first, in its order, then the outputs.
    assert columns[:3] == list(window._PEAK_TABLE_COLUMNS[:3])
    assert columns[3:] == ["Amplitude", "Area", "Area (%)"]
    assert table.rowCount() == 2
    # Numbered like the peak table, so row n here is row n there.
    assert [table.verticalHeaderItem(r).text() for r in range(2)] == ["1", "2"]

    shift_1 = table.item(0, 0).text()
    assert shift_1.startswith("3.0") and "\n± " in shift_1
    # Area share: the second peak is three times the first.
    shares = [float(table.item(r, 5).text()) for r in range(2)]
    assert abs(sum(shares) - 100.0) < 0.5
    assert abs(shares[1] / shares[0] - 3.0) < 0.1
    assert "RMSE" in window.spectral_fit_params_label.text()

    tsv = window._spectral_fit_table_as_tsv().splitlines()
    assert tsv[0].split("\t") == ["Peak"] + columns
    assert tsv[1].split("\t")[0] == "1"
    assert " ± " in tsv[1]  # one line per cell in the copy
    assert len(tsv) == 3

    window._clear_spectral_fit_report()
    assert table.rowCount() == 0
    assert window.spectral_fit_params_label.text() == ""


def test_spectral_fit_table_sits_in_the_isotropic_section(qapp):
    window = MainWindow()
    isotropic = next(s for s in window.findChildren(CollapsibleSection)
                     if s.title() == "Isotropic (Gauss/Lor)")
    assert isotropic.isAncestorOf(window.spectral_fit_table)
    assert isotropic.isAncestorOf(window.spectral_fit_params_label)


def test_value_error_formatting_keeps_only_supported_digits(qapp):
    fmt = MainWindow._format_value_error
    assert fmt(1.50234, 0.00031) == "1.50234 ± 0.00031"
    assert fmt(1.50234, 0.031) == "1.502 ± 0.031"
    assert fmt(1234.5, 12.0) == "1234 ± 12"   # banker's rounding of .5 either way is fine
    assert fmt(0.5, 0.0, fixed=True) == "0.5 (fixed)"
    assert fmt(2.0, float("nan")) == "2"
    assert fmt(2.0, None) == "2"


def test_value_error_formatting_rounds_to_tens_when_the_error_is_large(qapp):
    # Correction to the upstream formatter: with an error >= 100 it kept
    # every integer digit ("1234 ± 150"), more than the error supports.
    fmt = MainWindow._format_value_error
    assert fmt(1234.5, 150.0) == "1230 ± 150"
    assert fmt(98765.0, 4321.0) == "98800 ± 4300"
    assert fmt(-1234.5, 150.0) == "-1230 ± 150"
    assert fmt(55.0, 99.0) == "55 ± 99"          # boundary: still 0 decimals
    assert fmt(1234.0, 100.0) == "1230 ± 100"    # boundary: first rounded case


# ============================================================
# "Use the integration regions" -- ported from upstream (Leo, 14a6fd3).
# Any defined region used to restrict the fit silently; it is now one
# visible option, independent of the amplitude cap.
# ============================================================

def test_fit_uses_the_regions_only_as_the_two_boxes_say(qapp, monkeypatch):
    window = MainWindow()
    window._add_region_row(1.0, 2.0)
    window._add_peak_row(1.5, 0.1)
    spectrum = _make_1d_peak_spectrum(x0=1.5, fwhm=0.1)
    seen = {}

    def fake_fit(slices, peak_defs, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(processing_fitting, "NMR2D_fit_slices", fake_fit)

    def run():
        seen.clear()
        with pytest.raises(RuntimeError):
            window._fit_peaks([spectrum])
        return seen["fit_regions"], seen["amplitude_bound_regions"]

    assert window.fit_within_regions_checkbox.isChecked()  # the old behaviour, by default
    assert run() == ([(1.0, 2.0)], None)
    window.fit_within_regions_checkbox.setChecked(False)
    assert run() == (None, None)
    window.amplitude_bound_regions_checkbox.setChecked(True)
    assert run() == (None, [(1.0, 2.0)])
    window.fit_within_regions_checkbox.setChecked(True)
    assert run() == ([(1.0, 2.0)], [(1.0, 2.0)])


def test_region_options_are_visible_in_the_isotropic_section_not_advanced(qapp):
    # The whole point of the port: neither option may be hidden. Both
    # live in the Isotropic section's own "Use the integration regions"
    # group, outside the collapsed-by-default Advanced block.
    window = MainWindow()
    group = window.fit_regions_group
    assert group.isAncestorOf(window.fit_within_regions_checkbox)
    assert group.isAncestorOf(window.amplitude_bound_regions_checkbox)
    assert not window.peak_advanced_content.isAncestorOf(group)
    isotropic = next(s for s in window.findChildren(CollapsibleSection)
                     if s.title() == "Isotropic (Gauss/Lor)")
    assert isotropic.isAncestorOf(group)


def test_fit_within_regions_round_trips_through_reselection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window.fit_within_regions_checkbox.setChecked(False)
    window._save_current_analysis_state()

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    assert window.fit_within_regions_checkbox.isChecked()  # b: default (on)

    window._select_single_row(0)
    assert not window.fit_within_regions_checkbox.isChecked()


def test_fit_within_regions_defaults_on_for_a_state_saved_before_it_existed(qapp, tmp_path):
    # A state saved by an older build has no "fit_within_regions" key.
    # Those builds always fitted inside the regions, so that is what
    # restoring must give back -- not whatever the box currently shows.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._save_current_analysis_state()
    window.session.spectra[0].metadata.raw["analysis_state"].pop("fit_within_regions")

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window.fit_within_regions_checkbox.setChecked(False)
    window._select_single_row(0)
    assert window.fit_within_regions_checkbox.isChecked()


def test_no_region_edge_warning_when_regions_did_not_constrain_the_fit(qapp, monkeypatch):
    # The edge warning only means something when the fit window was the
    # region. With "Fit only inside the regions" off there is no edge,
    # so the check must return before even looking at the results.
    window = MainWindow()
    window._add_region_row(1.0, 2.0)
    window.fit_within_regions_checkbox.setChecked(False)

    def must_not_run(*a, **k):
        raise AssertionError("edge check ran although regions did not constrain the fit")

    monkeypatch.setattr(processing_fitting, "extract_params_and_errors", must_not_run)
    window._warn_if_fit_near_region_edge([{"popt": None, "pcov": None,
                                           "peak_defs": [], "is_free": None}])


def test_main_window_fit_spectrum_warns_when_result_lands_near_region_edge(qapp, monkeypatch):
    # Fitting-wizard roadmap item: "a sensible warning when a fit
    # result lands too close to its own region edge." Region (4.9,
    # 4.98) is deliberately narrower than the true peak needs (real
    # centre 5.0) -- the region masks out everything past 4.98, so the
    # fit can only see the peak's rising left shoulder and pushes the
    # fitted shift to the region's own upper edge, not because of the
    # position-tolerance bound (widened here specifically to rule that
    # out as the real constraint -- confirmed by direct reproduction
    # that a default-tolerance run hits *that* bound first instead,
    # which would defeat the purpose of this test).
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window.position_tolerance_spin.setValue(20.0)
    window._add_peak_row(shift=4.95, fwhm=0.05, eta=0.5)
    window._add_region_row(4.9, 4.98)

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window._on_fit_spectrum_clicked()

    assert len(warnings) == 1
    assert "peak 1" in warnings[0]
    assert "region" in warnings[0].lower()


def test_main_window_fit_spectrum_no_warning_when_peak_is_well_inside_region(qapp, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window.position_tolerance_spin.setValue(20.0)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    window._add_region_row(4.9, 5.5)  # peak sits centrally, nowhere near either edge

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window._on_fit_spectrum_clicked()

    assert warnings == []


def test_main_window_fit_spectrum_no_edge_warning_without_any_regions(qapp, monkeypatch):
    # No region at all -- an unconstrained fit has no "edge" to land
    # near, regardless of where the fitted shift ends up.
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    assert window.region_table.rowCount() == 0

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window._on_fit_spectrum_clicked()

    assert warnings == []


def test_main_window_fit_spectrum_not_supported_for_conventional_2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt out of the pseudo-2D default
    assert not window.session.spectra[window._selected_index()].is_pseudo_2d

    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))
    window._on_fit_spectrum_clicked()
    assert shown.get("called") is True


def test_main_window_fit_spectrum_requires_at_least_one_peak(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")

    shown = {}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("text", a[2]))
    window._on_fit_spectrum_clicked()
    assert "at least one peak" in shown.get("text", "")


class _FakeMenu:
    """``QMenu.exec()`` is modal and not interceptable headlessly by
    monkeypatching the real PySide6 class (confirmed the hard way --
    it just hangs) -- replace the name main_window.py actually calls
    (``QMenu(self)``) with this lightweight fake instead."""

    def __init__(self, *a, **k):
        self.added_labels = []

    def addAction(self, label):
        self.added_labels.append(label)
        return _FakeAction(label)

    def addSeparator(self):
        pass

    def exec(self, *a, **k):
        return None


def test_main_window_fit_curve_not_offered_for_1d_in_context_menu(qapp, tmp_path, monkeypatch):
    # Item C: curve fitting must only be offered (via context menu)
    # for a pseudo-2D spectrum or a Compare-tab stack -- not a plain
    # 1D spectrum.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()

    import opennmr.gui.main_window as main_window_module
    captured = []
    monkeypatch.setattr(main_window_module, "QMenu", lambda *a, **k: captured.append(_FakeMenu()) or captured[-1])

    pos = window.spectrum_list.visualItemRect(window.spectrum_list.item(index)).center()
    window._on_spectrum_list_context_menu(pos)

    assert not any("Fit Curve" in label for label in captured[0].added_labels)


def test_main_window_fit_curve_offered_for_pseudo2d_in_context_menu(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    index = window._selected_index()

    import opennmr.gui.main_window as main_window_module
    captured = []
    monkeypatch.setattr(main_window_module, "QMenu", lambda *a, **k: captured.append(_FakeMenu()) or captured[-1])

    pos = window.spectrum_list.visualItemRect(window.spectrum_list.item(index)).center()
    window._on_spectrum_list_context_menu(pos)

    assert any("Fit Curve" in label for label in captured[0].added_labels)


def test_main_window_fit_spectrum_then_curve_recovers_known_t2(qapp):
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 12)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=true_t2)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)  # eta=0 -> pure Gaussian, matches true shape
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"

    _fit_spectrum_then_curve(window)

    assert "T2" in window.curve_fit_summary_label.text()
    assert len(window.fit_canvas.ax.lines) == 2  # data points + fit curve
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_main_window_fit_canvas_siloed_per_spectrum(qapp):
    # Regression test: the Fit tab used to keep showing whichever
    # spectrum was last fit, even after switching to a different
    # spectrum with no fit of its own -- and switching back had lost
    # the original plot entirely, not just failed to clear it.
    delays = np.geomspace(0.001, 0.3, 12)
    spectrum_a = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)
    spectrum_b = _make_pseudo2d_relaxation_spectrum(delays, x0=3.0, fwhm=0.05, T2=0.08)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum_a)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"
    _fit_spectrum_then_curve(window)
    assert len(window.fit_canvas.ax.lines) == 2

    _load_spectrum_directly(window, spectrum_b)
    assert len(window.fit_canvas.ax.lines) == 0

    window._select_single_row(0)
    assert len(window.fit_canvas.ax.lines) == 2


def test_main_window_spectral_fit_canvas_siloed_per_spectrum(qapp):
    delays = np.geomspace(0.001, 0.3, 8)
    spectrum_a = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)
    spectrum_b = _make_pseudo2d_relaxation_spectrum(delays, x0=3.0, fwhm=0.05, T2=0.08)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum_a)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)
    window._on_fit_spectrum_clicked()  # spectral fit alone -- curve fit not needed here
    window.view_slice_fit_spin.setValue(2)  # look at one slice's spectral fit
    # Data + Fit (total) + baseline are lines; the single peak fitted
    # above is a shaded fill_between region (a collection, not a line).
    assert len(window.spectral_fit_canvas.ax.lines) == 3
    assert len(window.spectral_fit_canvas.ax.collections) == 1

    _load_spectrum_directly(window, spectrum_b)
    assert len(window.spectral_fit_canvas.ax.lines) == 0
    assert len(window.spectral_fit_canvas.ax.collections) == 0

    window._select_single_row(0)
    assert len(window.spectral_fit_canvas.ax.lines) == 3
    assert len(window.spectral_fit_canvas.ax.collections) == 1


def test_main_window_view_slice_fit_does_not_touch_spectrum_canvas(qapp):
    # Regression test for the reason this tab was split out: viewing a
    # slice's spectral fit used to overwrite the main Spectrum canvas,
    # losing whatever the user was looking at there.
    delays = np.geomspace(0.001, 0.3, 8)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)
    window._on_fit_spectrum_clicked()

    window.canvas_tabs.setCurrentWidget(window.canvas)
    spectrum_title_before = window.canvas.ax.get_title()

    window.view_slice_fit_spin.setValue(3)

    assert window.canvas.ax.get_title() == spectrum_title_before
    assert "Slice 3" not in window.canvas.ax.get_title()


def test_main_window_clear_resets_analysis_tab(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.5)
    window.analysis_results_label.setText("Peak 1: T2 = 1 ms")
    window.integration_results_label.setText("Integrated 1 slice(s) across 1 region(s)")

    window._on_clear_clicked()

    assert window.peak_table.rowCount() == 0
    assert window.analysis_results_label.text() == ""
    assert window.integration_results_label.text() == ""
    assert "pseudo-2D" in window.analysis_status_label.text()


# ============================================================
# Analysis tab restructure: Peak Picking / Integration split,
# Advanced (fit tolerance/S-N) collapsed by default
# ============================================================

def _enclosing_collapsible_section(widget):
    """The nearest :class:`CollapsibleSection` ancestor of ``widget``
    -- lets a test confirm two controls live in genuinely separate
    collapsible sections of a launcher panel (Fitting's Isotropic/
    Integration/CSA-Dipolar, etc.) without depending on exactly how
    each section's own internal layout is built."""
    node = widget.parentWidget()
    while node is not None and not isinstance(node, CollapsibleSection):
        node = node.parentWidget()
    return node


def test_main_window_isotropic_integration_csa_are_separate_sections(qapp):
    # Explicit user feedback: the combined "Spectral Fitting" sub-tab
    # held peak-picking, integration, and fit-bounds tuning all at
    # once -- "too much information at once for casual users". Peak
    # Picking (renamed "Isotropic (Gauss/Lor)"), Integration and CSA
    # (renamed "CSA/Dipolar") are now three independent collapsible
    # sections of the "Fitting" top-level tab -- see
    # test_main_window_fitting_tab_has_three_collapsible_sections for
    # the section structure itself.
    window = MainWindow()
    # VOCS moved to its own top-level "Advanced Processing" button;
    # Peak Picking, Integration and CSA all moved to the new
    # "Fitting" button; Curve Fitting to its own "Relaxation" button;
    # eNMR and Imaging to their own "Applications" button -- none of
    # these five belong on the old combined "Analysis" tab any more,
    # and that tab (and its analysis_sub_tabs) no longer exists at all.
    side = _side_tab_names(window)
    assert "Analysis" not in side
    assert not hasattr(window, "analysis_sub_tabs")

    peak_picking_section = _enclosing_collapsible_section(window.peak_table)
    integration_section = _enclosing_collapsible_section(window.region_table)
    csa_section = _enclosing_collapsible_section(window.csa_sites_table)
    assert peak_picking_section.title() == "Isotropic (Gauss/Lor)"
    assert integration_section.title() == "Integration"
    assert csa_section.title() == "CSA/Dipolar"
    # Three genuinely separate sections, not one shared container.
    assert len({id(peak_picking_section), id(integration_section), id(csa_section)}) == 3
    assert not peak_picking_section.isAncestorOf(window.region_table)
    assert not integration_section.isAncestorOf(window.peak_table)
    assert not csa_section.isAncestorOf(window.peak_table)


def test_main_window_peak_advanced_section_collapsed_by_default(qapp):
    # Position tolerance / S-N gating are fit-tuning knobs a casual
    # user's first fit never needs -- tucked behind a disclosure
    # toggle, not removed (same widgets/attributes as before, still
    # fully usable once expanded).
    window = MainWindow()
    assert window.peak_advanced_toggle.isChecked() is False
    assert window.peak_advanced_content.isHidden() is True

    window.peak_advanced_toggle.setChecked(True)

    assert window.peak_advanced_content.isHidden() is False
    # The actual tuning controls really are inside it, not just the
    # toggle button itself changing state independently.
    assert window.peak_advanced_content.isAncestorOf(window.position_tolerance_spin)
    assert window.peak_advanced_content.isAncestorOf(window.sn_noise_region_edit)
    assert window.peak_advanced_content.isAncestorOf(window.sn_threshold_spin)


def test_main_window_integration_results_persist_across_reselection(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="second1d"), vendor="bruker")
    first_index = 0
    window._on_selection_changed(first_index)
    window._add_region_row(1.0, 2.0)
    window.integration_results_label.setText("Integrated 1 slice(s) across 1 region(s)")
    window._save_current_analysis_state()

    window._on_selection_changed(1)
    assert window.integration_results_label.text() == ""  # a different spectrum's own state

    window._on_selection_changed(first_index)
    assert "Integrated" in window.integration_results_label.text()


# ============================================================
# Zero-fill dropdown (standard sizes, default = this spectrum's own
# actual current length -- not a "2x NP" suggestion)
# ============================================================

def test_main_window_zero_fill_defaults_to_spectrum_actual_length(qapp, tmp_path, monkeypatch):
    # Regression test: the combo used to default to a "2x NP, rounded
    # to the nearest preset" *suggestion* regardless of what the
    # vendor reader's own default zero-fill actually produced -- for
    # this fixture (procs SI == n, i.e. no zero-fill applied at load),
    # that meant the combo showed 2048 while the spectrum itself was
    # actually only 600 points. Any live recompute that reads this
    # combo (the Processing tab's own preview, and phase-dragging --
    # see _apply_phase_preview's "1d" branch) would then silently
    # zero-fill a freshly-loaded, never-reprocessed spectrum the
    # moment *any* control was touched. Fixed: default to reality.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=600), vendor="bruker")
    index = window._selected_index()
    assert len(window.session.spectra[index].data) == 600
    assert window._zero_fill_value() == 600
    assert window.zero_fill_custom_spin.isHidden() is False  # 600 isn't a standard preset -- correctly shows as Custom


def test_main_window_zero_fill_defaults_to_spectrum_actual_length_when_preset_sized(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=4096), vendor="bruker")
    assert window._zero_fill_value() == 4096
    assert window.zero_fill_custom_spin.isHidden() is True  # 4096 *is* a standard preset


def test_main_window_zero_fill_falls_back_to_custom_above_largest_preset(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=400_000), vendor="bruker")
    assert window._zero_fill_value() == 400_000
    assert window.zero_fill_custom_spin.isHidden() is False


def test_main_window_zero_fill_combo_selecting_custom_shows_spin(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    custom_index = window.zero_fill_combo.count() - 1
    assert window.zero_fill_combo.itemData(custom_index) is None

    window.zero_fill_combo.setCurrentIndex(custom_index)
    assert window.zero_fill_custom_spin.isHidden() is False
    window.zero_fill_custom_spin.setValue(12345)
    assert window._zero_fill_value() == 12345


def test_main_window_reprocess_uses_selected_preset_size(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import _ZERO_FILL_SIZES

    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()

    window._set_zero_fill_value(4096)
    window._on_fid_processing_control_changed()

    assert len(window.session.spectra[index].data) == 4096
    assert 4096 in _ZERO_FILL_SIZES


# ============================================================
# FID-processing metadata persistence across reselection
# ============================================================

def test_main_window_fid_processing_settings_persist_across_reselection(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")

    window._select_single_row(0)
    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)
    window.align_echo_button.setChecked(True)
    window.fid_shift_spin.setValue(2.5)
    window._set_zero_fill_value(4096)
    window._on_fid_processing_control_changed()

    # Switch away, then back -- controls must reflect what was
    # actually applied, not reset to the fresh-load defaults.
    window._select_single_row(1)
    window._select_single_row(0)

    assert window.window_combo.currentData() == "hann"
    assert window.align_echo_button.isChecked() is True
    assert window.fid_shift_spin.value() == pytest.approx(2.5)
    assert window._zero_fill_value() == 4096


def test_main_window_fid_processing_lb_persists_for_exponential_window(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")

    window._select_single_row(0)
    exp_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "exponential")
    window.window_combo.setCurrentIndex(exp_row)
    window.lb_spin.setValue(15.0)

    window._select_single_row(1)
    window._select_single_row(0)

    assert window.window_combo.currentData() == "exponential"
    assert window.lb_spin.value() == pytest.approx(15.0)


def test_main_window_changing_window_function_does_not_change_committed_phase(qapp, tmp_path, monkeypatch):
    # Real user concern: "make sure that changing the window function
    # doesn't reset the phase in any weird way (even just the output
    # spectrum)". Code reading shows phase0/phase1 fed into
    # processing_ft.apply_phase (both the single-row preview path,
    # _compute_processing_preview, and the multi-row path,
    # _recompute_multi_row_data) are computed purely from
    # spectrum.metadata.raw["phase0"/"phase1"] (+ the live drag delta
    # for the single-row case) -- no window-related term anywhere in
    # that expression, so they should be mathematically identical
    # regardless of which window is selected. Spying directly on the
    # actual (phase0, phase1) arguments passed to apply_phase is the
    # real check here -- comparing the *output spectrum's* phase at a
    # specific discrete sample instead (tried first) is confounded by
    # each window's own different lineshape/spectral-leakage shifting
    # exactly which sample is the magnitude-argmax, giving a
    # fractional-degree "difference" that's a measurement artefact of
    # a non-bin-exact synthetic peak, not a real phase change --
    # confirmed by hand before writing this test the more direct way.
    from opennmr.gui import main_window as main_window_module

    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256), vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    spectrum.metadata.raw["phase0"] = 45.0
    spectrum.metadata.raw["phase1"] = 5.0

    seen_phases = []
    real_apply_phase = main_window_module.processing_ft.apply_phase

    def spy_apply_phase(spectrum_arg, freq, phase0, phase1, **kwargs):
        seen_phases.append((phase0, phase1))
        return real_apply_phase(spectrum_arg, freq, phase0, phase1, **kwargs)

    monkeypatch.setattr(main_window_module.processing_ft, "apply_phase", spy_apply_phase)

    for key in ("exponential", "hann", "gaussian", "sine", "qsine"):
        row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == key)
        window.window_combo.setCurrentIndex(row)
        window._on_fid_processing_control_changed()

    assert len(seen_phases) >= 5  # at least once per window change
    assert all(p == pytest.approx((45.0, 5.0)) for p in seen_phases)


def test_main_window_gb_and_ssb_spins_enable_for_the_right_window_only(qapp, tmp_path, monkeypatch):
    # Real user report: Gaussian/Sine/QSine existed as window_combo
    # choices with no way to actually tune them (no ssb, no negative
    # lb, no gb) -- gb_spin/ssb_spin are new controls for exactly that;
    # confirm each is enabled only for the window shape it applies to.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._select_single_row(0)

    def set_window(key):
        row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == key)
        window.window_combo.setCurrentIndex(row)

    set_window("exponential")
    assert window.lb_spin.isEnabled() is True
    assert window.gb_spin.isEnabled() is False
    assert window.ssb_spin.isEnabled() is False

    set_window("gaussian")
    assert window.lb_spin.isEnabled() is True
    assert window.gb_spin.isEnabled() is True
    assert window.ssb_spin.isEnabled() is False

    set_window("sine")
    assert window.lb_spin.isEnabled() is False
    assert window.gb_spin.isEnabled() is False
    assert window.ssb_spin.isEnabled() is True

    set_window("qsine")
    assert window.ssb_spin.isEnabled() is True

    set_window("hann")
    assert window.lb_spin.isEnabled() is False
    assert window.gb_spin.isEnabled() is False
    assert window.ssb_spin.isEnabled() is False


def test_main_window_gaussian_window_reprocesses_without_crashing(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._select_single_row(0)
    index = window._selected_index()

    row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "gaussian")
    window.window_combo.setCurrentIndex(row)
    window.lb_spin.setValue(-5.0)
    window.gb_spin.setValue(20.0)
    window._on_fid_processing_control_changed()

    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[0] > 0
    assert np.all(np.isfinite(spectrum.data))


def test_main_window_qsine_window_reprocesses_without_crashing(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._select_single_row(0)
    index = window._selected_index()

    row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "qsine")
    window.window_combo.setCurrentIndex(row)
    window.ssb_spin.setValue(3.0)
    window._on_fid_processing_control_changed()

    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[0] > 0
    assert np.all(np.isfinite(spectrum.data))


def test_main_window_gb_and_ssb_persist_across_reselection(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")

    window._select_single_row(0)
    row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "gaussian")
    window.window_combo.setCurrentIndex(row)
    window.lb_spin.setValue(-3.0)
    window.gb_spin.setValue(12.0)
    window._on_fid_processing_control_changed()

    window._select_single_row(1)
    window._select_single_row(0)

    assert window.window_combo.currentData() == "gaussian"
    assert window.lb_spin.value() == pytest.approx(-3.0)
    assert window.gb_spin.value() == pytest.approx(12.0)


def test_main_window_unmodified_spectrum_still_gets_fresh_defaults(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="a"), vendor="bruker")
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="b"), vendor="bruker")

    window._select_single_row(0)
    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)

    # Spectrum 1 was never reprocessed -- selecting it must not show
    # spectrum 0's settings.
    window._select_single_row(1)
    assert window.window_combo.currentData() == "__recorded__"


# ============================================================
# Left/right splitter sizing
# ============================================================

def test_main_window_left_panel_has_a_usable_minimum_width(qapp):
    window = MainWindow()
    # The tiles are a persistent full-width row above both columns, so
    # the splitter is no longer the central widget -- it is held as
    # window.main_splitter instead.
    splitter = window.main_splitter
    assert isinstance(splitter, QSplitter)
    left_column = splitter.widget(0)
    assert left_column is window.left_splitter
    assert left_column.minimumWidth() >= 250
    assert window.side_tabs.minimumWidth() >= 250


# ============================================================
# Multi-canvas plot tabs (Spectrum / FID / Compare / Fit)
# ============================================================

def test_main_window_has_seven_canvas_tabs(qapp):
    window = MainWindow()
    labels = [window.canvas_tabs.tabText(i) for i in range(window.canvas_tabs.count())]
    assert labels == ["FID", "Processing Preview", "Spectrum", "Spectral Fit", "Fit", "Compare",
                      "Simulation"]
    assert window.canvas_tabs.widget(0) is window.fid_canvas
    assert window.canvas_tabs.widget(1) is window.processing_preview_canvas
    assert window.canvas_tabs.widget(2) is window.canvas
    # Fit/Compare/Processing Preview start hidden -- nothing to show
    # until a pseudo-2D spectrum is selected, the Compare list has an
    # entry, or a FID-reprocessable spectrum is selected, respectively.
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.fit_canvas)) is False
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.compare_canvas)) is False
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.processing_preview_canvas)) is False
    # Simulation: only once a Diffusion/Bloch simulation has run.
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.simulation_canvas)) is False


def test_main_window_selecting_spectrum_switches_to_spectrum_tab(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.canvas_tabs.currentWidget() is window.compare_canvas

    window._select_single_row(0)
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_clear_clears_all_six_canvases(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert _compare_line_count(window.compare_canvas) == 2

    window._on_clear_clicked()

    for canvas in (
        window.canvas, window.fid_canvas, window.processing_preview_canvas,
        window.compare_canvas, window.fit_canvas, window.spectral_fit_canvas,
    ):
        assert len(canvas.ax.lines) == 0


def test_main_window_2d_phase_drag_switches_to_spectrum_tab(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    two_d_index = window._selected_index()
    window._load(_make_1d_dataset(tmp_path, name="c1"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="c2"), vendor="bruker")
    _select_rows(window, [1, 2])  # the two 1D spectra -- jump to Compare tab
    assert window.canvas_tabs.currentWidget() is window.compare_canvas

    window._select_single_row(two_d_index)
    window.phase0_handle.setValue(20.0)
    assert window.canvas_tabs.currentWidget() is window.canvas


# ============================================================
# Pseudo-2D "Stacked slices" view
# ============================================================

def test_main_window_pseudo2d_view_combo_hidden_for_1d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.pseudo2d_view_combo.isHidden() is True


def test_main_window_pseudo2d_view_combo_visible_only_for_pseudo2d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    assert window.pseudo2d_view_combo.isHidden() is False  # pseudo-2D is the default

    window.process_indirect_button.setChecked(True)  # opt out -> conventional 2D
    assert window.pseudo2d_view_combo.isHidden() is True


def test_main_window_pseudo2d_stacked_view_shows_all_slices(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=5, n_f2=8), vendor="bruker")

    window.pseudo2d_view_combo.setCurrentIndex(1)  # Stacked slices

    assert _compare_line_count(window.canvas) == 5
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_pseudo2d_heatmap_view_renders(qapp, tmp_path):
    # Real roadmap ask: "pseudo-2D/stacked-1D data should be plottable
    # as either a contour or a heatmap." plot_1d_heatmap already
    # existed (ported from nmrtools2's own NMR1dheatmap), just never
    # called from GUI code.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=5, n_f2=8), vendor="bruker")

    window.pseudo2d_view_combo.setCurrentIndex(2)  # Heatmap

    assert window.canvas_tabs.currentWidget() is window.canvas
    assert window.canvas._contour_spectrum is None  # not the single-spectrum contour view
    assert len(window.canvas.ax.collections) >= 1  # pcolormesh


def test_main_window_pseudo2d_view_resets_to_contour_on_reselection(qapp, tmp_path):
    # Contour is the default view for a fresh pseudo-2D selection --
    # explicit user feedback: the Spectrum tab is also where phasing
    # happens, which needs a plain single-slice/contour view, not the
    # multi-axes Compare-style "Stacked slices" plot. Runs
    # unconditionally on every reselection, so a Stacked choice left
    # over from a previous spectrum is overridden, not carried forward.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="a"), vendor="bruker")
    window.pseudo2d_view_combo.setCurrentIndex(1)  # Stacked slices

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window._select_single_row(0)

    assert window.pseudo2d_view_combo.currentIndex() == 0


# ============================================================
# Diffusion (PFG) fitting
# ============================================================

def test_main_window_fit_type_defaults_to_relaxation_models(qapp):
    from opennmr.gui.main_window import _RELAXATION_MODEL_ITEMS

    window = MainWindow()
    keys = [window.relaxation_model_combo.itemData(i) for i in range(window.relaxation_model_combo.count())]
    assert keys == [key for _label, key in _RELAXATION_MODEL_ITEMS] + ["custom"]


def test_main_window_fit_type_diffusion_switches_model_list_and_label(qapp):
    from opennmr.gui.main_window import _DIFFUSION_MODEL_ITEMS

    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)

    keys = [window.relaxation_model_combo.itemData(i) for i in range(window.relaxation_model_combo.count())]
    assert keys == [key for _label, key in _DIFFUSION_MODEL_ITEMS] + ["custom"]
    assert "b-values" in window.delays_row_label.text()

    window.fit_type_combo.setCurrentIndex(0)  # back to Relaxation
    assert "Delays" in window.delays_row_label.text()


def test_main_window_resolve_b_values_manual_entry(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    window.manual_delays_edit.setText("1e8, 5e8, 2e9")
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])  # vendor/source_path irrelevant here

    b_values = window._resolve_independent_values(spectrum, is_diffusion=True)
    np.testing.assert_allclose(b_values, [1e8, 5e8, 2e9])


def test_main_window_resolve_b_values_auto_falls_back_to_slice_index(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    b_values = window._resolve_independent_values(spectrum, is_diffusion=True)
    np.testing.assert_allclose(b_values, [1.0, 2.0, 3.0])


def test_main_window_diffusion_fit_recovers_known_d(qapp):
    from opennmr.processing.relaxation import stejskal_tanner_decay

    true_D = 2.0e-9
    b_values = np.linspace(0.0, 3.0e9, 14)
    ppm = np.linspace(3.0, 7.0, 800)
    x0, fwhm = 5.0, 0.05
    data = np.zeros((len(b_values), len(ppm)))
    for i, b in enumerate(b_values):
        amplitude = stejskal_tanner_decay(b, 1.0, true_D)
        data[i] = gaussian(ppm, x0, fwhm, amplitude)
    f1_axis = NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    spectrum = NMRSpectrum(
        data=data, axes=[f1_axis, f2_axis],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=x0, fwhm=fwhm, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    window.manual_delays_edit.setText(", ".join(f"{b:.6g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)  # Stejskal-Tanner

    _fit_spectrum_then_curve(window)

    assert "D" in window.curve_fit_summary_label.text()
    assert "m^2/s" in window.curve_fit_summary_label.text() or "e-" in window.curve_fit_summary_label.text()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert window.fit_canvas.ax.get_xlabel() == "b-value (s/m²)"


def test_main_window_biexponential_diffusion_fit_recovers_known_ds(qapp):
    # Multicomponent PFG fitting -- explicit user feedback asking for
    # this specifically. rate_param is None for this model (two D
    # values, not one), so the results label falls back to showing
    # every fitted parameter rather than a single "D = ..." summary.
    from opennmr.processing.relaxation import stejskal_tanner_decay_biexponential

    I0a_true, Da_true, I0b_true, Db_true = 0.4, 0.5e-9, 0.6, 5.0e-9
    b_values = np.linspace(0.0, 4.0e9, 20)
    ppm = np.linspace(3.0, 7.0, 800)
    x0, fwhm = 5.0, 0.05
    data = np.zeros((len(b_values), len(ppm)))
    for i, b in enumerate(b_values):
        amplitude = stejskal_tanner_decay_biexponential(b, I0a_true, Da_true, I0b_true, Db_true)
        data[i] = gaussian(ppm, x0, fwhm, amplitude)
    f1_axis = NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    spectrum = NMRSpectrum(
        data=data, axes=[f1_axis, f2_axis],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=x0, fwhm=fwhm, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    window.manual_delays_edit.setText(", ".join(f"{b:.6g}" for b in b_values))
    biexp_index = next(
        i for i in range(window.relaxation_model_combo.count())
        if window.relaxation_model_combo.itemData(i) == "stejskal_tanner_biexponential"
    )
    window.relaxation_model_combo.setCurrentIndex(biexp_index)

    _fit_spectrum_then_curve(window)

    results_text = window.curve_fit_summary_label.text()
    assert "Da" in results_text and "Db" in results_text
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


# ============================================================
# Analysis-tab state siloed per spectrum
# ============================================================

def test_main_window_peak_list_siloed_per_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="a"), vendor="bruker")
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.5)
    assert window.peak_table.rowCount() == 1

    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="b"), vendor="bruker")
    assert window.peak_table.rowCount() == 0  # a fresh spectrum starts empty, not lingering

    window._add_peak_row(shift=6.0, fwhm=0.2, eta=0.3)
    window._select_single_row(0)
    assert window.peak_table.rowCount() == 1
    assert window.peak_table.item(0, 0).text() == "5"

    window._select_single_row(1)
    assert window.peak_table.rowCount() == 1
    assert window.peak_table.item(0, 0).text() == "6"


def test_main_window_fit_type_and_delay_source_siloed_per_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="a"), vendor="bruker")
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion
    window.delay_source_combo.setCurrentIndex(2)  # Start + step
    window.delay_start_spin.setValue(1.0)

    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="b"), vendor="bruker")
    assert window.fit_type_combo.currentIndex() == 0  # fresh spectrum -> back to defaults
    assert window.delay_source_combo.currentIndex() == 0

    window._select_single_row(0)
    assert window.fit_type_combo.currentIndex() == 1
    assert window.delay_source_combo.currentIndex() == 2
    assert window.delay_start_spin.value() == pytest.approx(1.0)


def test_main_window_analysis_results_siloed_per_spectrum(qapp):
    delays = np.geomspace(0.001, 0.3, 10)
    spectrum_a = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)
    spectrum_b = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum_a)
    _load_spectrum_directly(window, spectrum_b)

    window._select_single_row(0)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)
    _fit_spectrum_then_curve(window)
    assert window.curve_fit_summary_label.text() != ""

    window._select_single_row(1)
    assert window.curve_fit_summary_label.text() == ""  # never fitted -- not spectrum a's result
    assert window.view_slice_fit_spin.isEnabled() is False

    window._select_single_row(0)
    assert window.curve_fit_summary_label.text() != ""
    assert window.view_slice_fit_spin.isEnabled() is True


# ============================================================
# Spectral (per-slice) fit browsing
# ============================================================

def test_main_window_view_slice_fit_disabled_before_fit(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    assert window.view_slice_fit_spin.isEnabled() is False


def test_main_window_view_slice_fit_shows_data_and_fit_curve(qapp):
    delays = np.geomspace(0.001, 0.3, 8)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")  # these delays are in seconds
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)
    window._on_fit_spectrum_clicked()  # spectral fit alone -- curve fit not needed here

    assert window.view_slice_fit_spin.isEnabled() is True
    assert window.view_slice_fit_spin.maximum() == len(delays)

    window.view_slice_fit_spin.setValue(3)
    assert window.canvas_tabs.currentWidget() is window.spectral_fit_canvas
    ax = window.spectral_fit_canvas.ax
    # The single peak is a shaded fill_between region (a collection),
    # not a line -- data/baseline/total fit remain plain lines.
    labels = [artist.get_label() for artist in (*ax.lines, *ax.collections)]
    assert "Data" in labels and "Fit (total)" in labels
    assert any(label.startswith("Peak 1") for label in labels)
    assert "Baseline" in labels
    assert "Slice 3" in window.spectral_fit_canvas.ax.get_title()
    # Results are a table now (one row per peak), not a text line.
    assert window.spectral_fit_table.rowCount() == 1
    assert window.spectral_fit_table.verticalHeaderItem(0).text() == "1"

    # Explicit user request: left/right buttons, matching the pattern
    # already used for View curve fit (peak #) and the phasing-tab
    # slice picker.
    window.slice_fit_next_button.click()
    assert window.view_slice_fit_spin.value() == 4
    window.slice_fit_prev_button.click()
    window.slice_fit_prev_button.click()
    assert window.view_slice_fit_spin.value() == 2


def test_main_window_view_curve_fit_row_sits_below_fit_curve_settings(qapp):
    # Explicit user feedback: "View curve fit" (formerly "View peak
    # fit") browses Fit Curve's own result, so it should sit below the
    # button that produces it, not above -- it used to sit between
    # Peaks and Fit Spectrum's own result, implying (wrongly) that it
    # was part of step 1.
    window = MainWindow()
    fit_curve_settings_box = window.fit_type_combo.parentWidget()
    tab = window.relaxation_uncertainty_checkbox.parentWidget()
    assert tab is fit_curve_settings_box.parentWidget()  # both direct children of the same tab layout
    layout = tab.layout()
    assert layout.indexOf(fit_curve_settings_box) < layout.indexOf(window.relaxation_uncertainty_checkbox)


def test_main_window_analysis_tab_labels_renamed_for_clarity(qapp):
    # Explicit user feedback: "slice fit"/"peak fit" read as
    # confusingly similar -- renamed to make clear one is a spectral
    # (per-slice) fit and the other a curve (per-peak) fit.
    # The two now live in different *top-level* tabs -- the spectral
    # (per-slice) one on Fitting, the curve (per-peak) one on
    # Relaxation -- so search both rather than one sub-tab widget. The
    # point of this test is the wording, which is unchanged.
    window = MainWindow()
    side = _side_tab_names(window)
    fitting_tab = window.side_tabs.widget(window._fitting_tab_index)
    relaxation_tab = window.side_tabs.widget(side.index("Relaxation"))
    wanted = ("Spectrum fit (slice #):", "View curve fit (peak #):")
    labels = [
        child.text()
        for parent in (fitting_tab, relaxation_tab)
        for child in parent.findChildren(QLabel) if child.text() in wanted
    ]
    assert set(labels) == set(wanted)


def test_main_window_peak_table_height_is_capped(qapp):
    # Explicit user feedback: the left side panel needs scrolling too
    # often -- Qt's own default QTableWidget size hint reserves
    # generous vertical space regardless of how many peaks are
    # actually in it (typically a handful), which was the single
    # biggest contributor to the Analysis tab needing more height than
    # even a modest (not tiny) window comfortably provides.
    window = MainWindow()
    assert window.peak_table.maximumHeight() <= 160


def test_main_window_fit_spectrum_updates_peak_table_starting_values(qapp):
    delays = np.geomspace(0.001, 0.3, 8)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.2, fwhm=0.08, eta=0.3)  # deliberately off from the true 5.0/0.05
    window.position_tolerance_spin.setValue(5.0)  # wide enough that the default +/-0.5*FWHM bound doesn't clip the fit short of 5.0
    window._on_fit_spectrum_clicked()

    shift = float(window.peak_table.item(0, 0).text())
    fwhm = float(window.peak_table.item(0, 1).text())
    assert abs(shift - 5.0) < 0.02
    assert abs(fwhm - 0.05) < 0.02


# ============================================================
# Spectral reference (SR) correction
# ============================================================

def test_main_window_apply_sr_shifts_direct_axis(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    original_ppm = spectrum.axes[-1].ppm.copy()
    original_hz = spectrum.axes[-1].hz.copy()
    sfo1 = spectrum.metadata.sfo1

    delta_hz = 0.2 * sfo1
    window.sr_shift_spin.setValue(delta_hz)
    window._on_apply_sr_clicked()

    np.testing.assert_allclose(spectrum.axes[-1].ppm, original_ppm + 0.2)
    np.testing.assert_allclose(spectrum.axes[-1].hz, original_hz + delta_hz)


def test_main_window_apply_sr_zero_delta_is_noop(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()
    original_ppm = window.session.spectra[index].axes[-1].ppm.copy()

    window.sr_shift_spin.setValue(0.0)
    window._on_apply_sr_clicked()

    np.testing.assert_allclose(window.session.spectra[index].axes[-1].ppm, original_ppm)


def test_main_window_apply_sr_does_not_reset_control(qapp, tmp_path):
    # SR is a persistent value, not a one-shot delta -- the control
    # must keep showing what was applied, not snap back to 0.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.sr_shift_spin.setValue(40.0)
    window._on_apply_sr_clicked()
    assert window.sr_shift_spin.value() == pytest.approx(40.0)


def test_main_window_apply_sr_is_idempotent(qapp, tmp_path):
    # Applying "the same SR" a second time must be a no-op, not a
    # second shift on top of the first (the old additive-delta design
    # would have doubled it).
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    window.sr_shift_spin.setValue(40.0)
    window._on_apply_sr_clicked()
    once = spectrum.axes[-1].hz.copy()

    window._on_apply_sr_clicked()  # re-apply the same value
    np.testing.assert_allclose(spectrum.axes[-1].hz, once)


def test_main_window_apply_sr_second_value_replaces_not_compounds(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    original_hz = spectrum.axes[-1].hz.copy()

    window.sr_shift_spin.setValue(40.0)
    window._on_apply_sr_clicked()
    window.sr_shift_spin.setValue(10.0)
    window._on_apply_sr_clicked()

    np.testing.assert_allclose(spectrum.axes[-1].hz, original_hz + 10.0)


def test_main_window_sr_siloed_per_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window.sr_shift_spin.setValue(40.0)
    window._on_apply_sr_clicked()

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    assert window.sr_shift_spin.value() == pytest.approx(0.0)  # fresh spectrum -> no SR yet

    window._select_single_row(0)
    assert window.sr_shift_spin.value() == pytest.approx(40.0)


def test_main_window_apply_sr_preserves_axis_unit(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.canvas.set_axis_unit("hz")

    window.sr_shift_spin.setValue(40.0)
    window._on_apply_sr_clicked()

    assert window.canvas.axis_unit == "hz"
    assert window.canvas.ax.get_xlabel() == "Frequency (Hz)"


def test_main_window_reference_shift_from_plot_prompts_and_applies(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    original_hz = spectrum.axes[-1].hz.copy()

    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (50.0, True))
    window._on_reference_shift_from_plot()

    np.testing.assert_allclose(spectrum.axes[-1].hz, original_hz + 50.0)
    assert spectrum.metadata.raw["sr_shift_hz"] == pytest.approx(50.0)


def test_main_window_reference_shift_from_plot_cancelled_is_noop(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    index = window._selected_index()
    original_hz = window.session.spectra[index].axes[-1].hz.copy()

    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (50.0, False))
    window._on_reference_shift_from_plot()

    np.testing.assert_allclose(window.session.spectra[index].axes[-1].hz, original_hz)


# ============================================================
# RelativeDoubleSpinBox: typed arithmetic deltas ("45+90" -> 135)
# ============================================================

def test_relative_spinbox_addition_expression(qapp):
    from opennmr.gui.widgets.relative_spinbox import RelativeDoubleSpinBox
    spin = RelativeDoubleSpinBox()
    assert spin.valueFromText("45+90") == pytest.approx(135.0)


def test_relative_spinbox_subtraction_expression(qapp):
    from opennmr.gui.widgets.relative_spinbox import RelativeDoubleSpinBox
    spin = RelativeDoubleSpinBox()
    assert spin.valueFromText("100-30") == pytest.approx(70.0)


def test_relative_spinbox_negative_operands(qapp):
    from opennmr.gui.widgets.relative_spinbox import RelativeDoubleSpinBox
    spin = RelativeDoubleSpinBox()
    assert spin.valueFromText("-45+90") == pytest.approx(45.0)
    assert spin.valueFromText("45+-90") == pytest.approx(-45.0)


def test_relative_spinbox_plain_number_still_works(qapp):
    from opennmr.gui.widgets.relative_spinbox import RelativeDoubleSpinBox
    spin = RelativeDoubleSpinBox()
    assert spin.valueFromText("45") == pytest.approx(45.0)


def test_relative_spinbox_bare_negative_is_literal_not_relative(qapp):
    # A bare "-45" (no second operand) must set the literal value -45,
    # not be misread as a relative delta -- selecting all and typing a
    # negative number is completely normal and must not be ambiguous.
    from opennmr.gui.widgets.relative_spinbox import RelativeDoubleSpinBox
    spin = RelativeDoubleSpinBox()
    spin.setRange(-180.0, 180.0)
    assert spin.valueFromText("-45") == pytest.approx(-45.0)


def test_relative_spinbox_strips_suffix_before_parsing(qapp):
    from opennmr.gui.widgets.relative_spinbox import RelativeDoubleSpinBox
    spin = RelativeDoubleSpinBox()
    spin.setSuffix(" ppm")
    assert spin.valueFromText("5+2.5 ppm") == pytest.approx(7.5)


def test_main_window_phase0_spin_accepts_relative_expression(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    window.phase0_spin.setValue(45.0)
    window.phase0_spin.setValue(window.phase0_spin.valueFromText("45+90"))
    assert window.phase0_spin.value() == pytest.approx(135.0)


def test_main_window_pivot_spin_accepts_relative_expression(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    window.pivot_spin.setValue(window.pivot_spin.valueFromText("1+2"))
    assert window.pivot_spin.value() == pytest.approx(3.0)


# ============================================================
# VOCS and spectrum arithmetic (Phase 1 backend-exposure sweep)
# ============================================================

def _make_1d_peak_spectrum_with_pulse(x0=0.0, fwhm=40.0, amplitude=1.0, n_pts=400, p1=5.0, sfo1=400.0):
    # Deliberately WIDE relative to the pulse's own excitation
    # bandwidth (w1 = 50 kHz for p1 = 5 us, against a +/-400 kHz
    # sweep), because that is the regime VOCS exists for. The old
    # fixture spanned 3-7 ppm = 1600 Hz, over which the excitation
    # profile is 1.0 to four decimal places -- so a *correct*
    # correction is indistinguishable from a no-op there, and the only
    # reason the old test saw a change at all was the x2 bug.
    ppm = np.linspace(x0 - 1000.0, x0 + 1000.0, n_pts)
    data = gaussian(ppm, x0, fwhm, amplitude)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    # `P` as a real indexed array, matching an actual acqus -- see
    # tests/test_vocs.py's own fixture for why that matters.
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1,
                            raw={"acqus": {"P": [0.0, p1, 2 * p1] + [0.0] * 61}})
    return NMRSpectrum(data=data, axes=[axis], metadata=metadata)


def test_main_window_apply_vocs_adds_derived_spectrum(qapp):
    window = MainWindow()
    spectrum = _make_1d_peak_spectrum_with_pulse()
    _load_spectrum_directly(window, spectrum)

    window._on_apply_vocs_clicked()

    assert window.spectrum_list.count() == 2
    derived = window.session.spectra[1]
    assert derived.metadata.vendor == "derived"
    assert not np.allclose(derived.data, spectrum.data)
    # Weighting by the excitation profile can only ever attenuate --
    # the weight is <= 1 everywhere, and exactly 1 on resonance, so
    # the scale is preserved rather than amplified. A divide-by-the-
    # profile deconvolution was tried here and reverted.
    nonzero = spectrum.data != 0
    gain = derived.data[nonzero] / spectrum.data[nonzero]
    assert gain.max() <= 1.0 + 1e-12


def test_main_window_check_all_toggles_every_compare_row_at_once(qapp):
    # "Make it easier to select all and VOCS instead of one at a time"
    # -- checking a dozen offsets individually is the slow path
    # everything downstream (VOCS Sum, Fit Curve, the plot) depends on.
    window = MainWindow()
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=offset))
    window.spectrum_list.selectAll()

    window._set_all_compare_checked(False)
    assert window._compare_checked_1d_stack() == []

    window._set_all_compare_checked(True)
    assert len(window._compare_checked_1d_stack()) == window.compare_list.count()


def test_main_window_check_all_replots_once_not_per_row(qapp):
    # compare_list.itemChanged triggers a full replot, so toggling n
    # rows individually would rebuild the figure n times.
    window = MainWindow()
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=offset))
    window.spectrum_list.selectAll()

    calls = []
    original = window._refresh_compare_view_if_active
    window._refresh_compare_view_if_active = lambda: (calls.append(1), original())[1]
    try:
        window._set_all_compare_checked(True)
    finally:
        window._refresh_compare_view_if_active = original

    assert calls == [1]


def test_main_window_apply_vocs_no_selection_shows_message(qapp, monkeypatch):
    window = MainWindow()
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_apply_vocs_clicked()

    assert shown.get("called") is True
    assert window.spectrum_list.count() == 0


def test_main_window_apply_vocs_bad_pulse_shows_error(qapp, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    window.vocs_pulse_edit.setText("P9")
    shown = {}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))

    window._on_apply_vocs_clicked()

    assert shown.get("called") is True
    assert window.spectrum_list.count() == 1


def test_main_window_vocs_sum_combines_compare_stack(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=4.8))
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=5.2))
    _select_rows(window, [0, 1])

    window._on_vocs_sum_clicked()

    assert window.spectrum_list.count() == 3
    derived = window.session.spectra[2]
    assert derived.metadata.vendor == "derived"
    assert derived.ndim == 1


def test_main_window_vocs_sum_requires_checked_entries(qapp, monkeypatch):
    window = MainWindow()
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_vocs_sum_clicked()

    assert shown.get("called") is True
    assert window.spectrum_list.count() == 0


def test_main_window_combine_spectra_sum(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=4.8))
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=5.2))
    _select_rows(window, [0, 1])
    window.combine_op_combo.setCurrentIndex(window.combine_op_combo.findData("sum"))

    window._on_combine_spectra_clicked()

    assert window.spectrum_list.count() == 3
    derived = window.session.spectra[2]
    assert derived.metadata.vendor == "derived"
    assert derived.ndim == 1


def test_main_window_combine_spectra_average_differs_from_sum(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=4.8))
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=5.2))
    _select_rows(window, [0, 1])

    window.combine_op_combo.setCurrentIndex(window.combine_op_combo.findData("sum"))
    window._on_combine_spectra_clicked()
    summed = window.session.spectra[2]

    window.combine_op_combo.setCurrentIndex(window.combine_op_combo.findData("average"))
    window._on_combine_spectra_clicked()
    averaged = window.session.spectra[3]

    assert np.allclose(averaged.data * 2.0, summed.data)


def test_main_window_combine_spectra_requires_two(qapp, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_combine_spectra_clicked()

    assert shown.get("called") is True
    assert window.spectrum_list.count() == 1


# ============================================================
# Output tab (Phase 4: live style panel, export settings, per-type templates)
# ============================================================

def test_main_window_has_no_output_tab_any_more(qapp):
    # Explicit user request: the Plot Style dialog "supplants the need
    # for an output tab". The controls survive as the dialog's body --
    # same widget objects, so every handler still reads self.output_*.
    window = MainWindow()
    names = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    assert "Output" not in names
    assert window.output_linewidth_spin is not None
    assert window._style_panel is not None


def test_main_window_plot_style_dialog_opens_on_the_current_plot(qapp):
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    window.compare_canvas.set_style(linewidth=4.25)

    window._on_plot_style_clicked()

    assert window._style_dialog is not None
    assert window.output_linewidth_spin.value() == pytest.approx(4.25)
    assert "Compare" in window.style_scope_label.text()
    window._style_dialog.close()


def test_main_window_plot_style_is_reachable_from_the_canvas(qapp):
    # Asked for as "a plot view popup (from right click menu or top
    # menu)" -- both routes reach the same dialog.
    window = MainWindow()
    window.canvas.plotStyleRequested.emit()
    assert window._style_dialog is not None
    window._style_dialog.close()


def test_main_window_trace_colour_applies_to_the_spectrum_tab(qapp, tmp_path):
    # Explicit user request: "the colours should apply to the spectrum
    # tab as well as the compare tab". plot_1d always took a colour;
    # nothing ever passed one.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.canvas)

    window.canvas.set_style(trace_color="#cc3311")

    assert window.canvas.style_settings()["trace_color"] == "#cc3311"
    trace = _trace_lines(window.canvas.ax)[0]
    assert matplotlib.colors.to_hex(trace.get_color()) == "#cc3311"
    # And it survives a redraw, not just the in-place recolour.
    window.canvas.show_spectrum(window.session.spectra[0])
    assert matplotlib.colors.to_hex(
        _trace_lines(window.canvas.ax)[0].get_color()) == "#cc3311"


def test_main_window_rename_reaches_every_place_a_spectrum_is_named(qapp, tmp_path, monkeypatch):
    # One override, honoured by Session.label, rather than a separate
    # editable field per view.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("My sample", True))

    window._on_rename_spectrum_action(0)

    assert window.session.label(0) == "My sample"
    assert window.spectrum_list.item(0).text() == "My sample"

    # Blank restores the derived name.
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("   ", True))
    window._on_rename_spectrum_action(0)
    assert window.session.label(0) != "My sample"


def test_main_window_saved_style_defaults_are_applied_at_startup(qapp, tmp_path, monkeypatch):
    config = tmp_path / "plot_style.json"
    config.write_text(json.dumps({"linewidth": 2.75, "labelsize": 19, "grid": True}))
    monkeypatch.setattr(MainWindow, "_PLOT_STYLE_CONFIG_PATH", str(config))

    window = MainWindow()

    for canvas in window._all_canvases():
        assert canvas._AXIS_LINEWIDTH == pytest.approx(2.75)
        assert canvas._AXIS_LABELSIZE == 19
        assert canvas._grid_visible is True


def test_main_window_missing_style_defaults_are_not_an_error(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_PLOT_STYLE_CONFIG_PATH", str(tmp_path / "nope.json"))
    window = MainWindow()          # must not raise
    assert window.canvas._AXIS_LINEWIDTH == pytest.approx(MplCanvas._AXIS_LINEWIDTH)


def test_main_window_save_style_default_writes_and_applies(qapp, tmp_path, monkeypatch):
    config = tmp_path / "plot_style.json"
    monkeypatch.setattr(MainWindow, "_PLOT_STYLE_CONFIG_PATH", str(config))
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.output_linewidth_spin.setValue(3.5)

    window._on_save_style_default_clicked()

    assert json.loads(config.read_text())["linewidth"] == pytest.approx(3.5)
    # Saving a default also puts it everywhere now, not just on disk.
    for canvas in window._all_canvases():
        assert canvas._AXIS_LINEWIDTH == pytest.approx(3.5)


def test_main_window_style_change_applies_to_the_plot_you_are_on(qapp):
    # Style used to be fanned out to every canvas on every change.
    # It is now per plot -- which is what makes "Apply to all" a real
    # action, and what lets a Compare stack be styled differently from
    # the Spectrum tab. Explicit user request.
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.output_linewidth_spin.setValue(3.0)

    assert window.canvas._AXIS_LINEWIDTH == pytest.approx(3.0)
    assert window.compare_canvas._AXIS_LINEWIDTH != pytest.approx(3.0)


def test_main_window_apply_style_to_all_pushes_it_everywhere(qapp):
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.output_linewidth_spin.setValue(3.0)

    window._on_apply_style_to_all_clicked()

    for canvas in window._all_canvases():
        assert canvas._AXIS_LINEWIDTH == pytest.approx(3.0)


def test_main_window_style_controls_reload_from_the_plot_being_edited(qapp):
    # Opening the dialog on another plot has to show *that* plot's
    # settings, or the first edit silently overwrites them.
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.output_linewidth_spin.setValue(3.0)

    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    window._load_style_controls_from(window.compare_canvas)
    assert window.output_linewidth_spin.value() == pytest.approx(
        window.compare_canvas._AXIS_LINEWIDTH)

    window._load_style_controls_from(window.canvas)
    assert window.output_linewidth_spin.value() == pytest.approx(3.0)


def test_main_window_axis_label_style_combo_changes_the_spectrum_label(qapp, tmp_path):
    """Explicit user request: "add Arial as the default font... [and]
    the axis labels should also be... brackets [not] slash... People
    should be allowed to swap between these". The panel's own combo
    box is the only way a person actually reaches this -- exercising
    canvas.set_style directly (see test_gui_canvas.py) doesn't confirm
    the widget is wired to it at all."""
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.canvas)

    label = window.canvas.ax.get_xlabel()
    assert r"\delta" in label and "1" in label and "H" in label and " / " not in label

    slash_index = window.output_axis_label_style_combo.findData("slash")
    window.output_axis_label_style_combo.setCurrentIndex(slash_index)
    assert " / " in window.canvas.ax.get_xlabel()

    window.output_axis_label_template_edit.setText("{element}-{mass} shift ({unit})")
    window.output_axis_label_template_edit.editingFinished.emit()
    assert window.canvas.ax.get_xlabel() == "H-1 shift (ppm)"

    # Blank restores the style combo's own choice (still "slash" here).
    window.output_axis_label_template_edit.setText("")
    window.output_axis_label_template_edit.editingFinished.emit()
    assert "/" in window.canvas.ax.get_xlabel()
    assert "shift" not in window.canvas.ax.get_xlabel()


def test_main_window_saved_axis_label_settings_are_applied_at_startup(qapp, tmp_path, monkeypatch):
    config = tmp_path / "plot_style.json"
    config.write_text(json.dumps({
        "axis_label_style": "slash",
        "axis_label_template": "{element}-{mass} ({unit})",
        "font_family": "monospace",
    }))
    monkeypatch.setattr(MainWindow, "_PLOT_STYLE_CONFIG_PATH", str(config))

    window = MainWindow()

    assert window.output_axis_label_style_combo.currentData() == "slash"
    assert window.output_axis_label_template_edit.text() == "{element}-{mass} ({unit})"
    assert window.output_font_family_combo.currentData() == "monospace"
    for canvas in window._all_canvases():
        assert canvas._axis_label_template == "{element}-{mass} ({unit})"
        assert canvas._font_family == "monospace"


def test_main_window_save_style_default_writes_axis_label_settings(qapp, tmp_path, monkeypatch):
    config = tmp_path / "plot_style.json"
    monkeypatch.setattr(MainWindow, "_PLOT_STYLE_CONFIG_PATH", str(config))
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    slash_index = window.output_axis_label_style_combo.findData("slash")
    window.output_axis_label_style_combo.setCurrentIndex(slash_index)

    window._on_save_style_default_clicked()

    saved = json.loads(config.read_text())
    assert saved["axis_label_style"] == "slash"
    for canvas in window._all_canvases():
        assert canvas._axis_label_style == "slash"


def test_main_window_contour_fraction_spin_defaults_to_25_percent(qapp):
    # Matches plotting.plot_2d_contour's own default (0.25) -- explicit
    # user request: "a factor of 4 less should be appropriate" as the
    # default, made user-adjustable rather than hardcoded.
    window = MainWindow()
    assert window.output_contour_fraction_spin.value() == 25
    assert window.canvas._contour_fraction == pytest.approx(0.25)


def test_main_window_contour_fraction_change_applies_to_every_canvas(qapp):
    window = MainWindow()
    window.output_contour_fraction_spin.setValue(60)
    for canvas in window._all_canvases():
        assert canvas._contour_fraction == pytest.approx(0.6)


def test_main_window_nq_line_checkbox_defaults_off(qapp):
    window = MainWindow()
    assert window.output_nq_line_checkbox.isChecked() is False
    assert window.output_nq_value_spin.isEnabled() is False
    assert window.canvas._show_nq_line is False
    assert window.canvas._nq_values == [1]


def test_main_window_nq_line_toggle_applies_to_every_canvas(qapp):
    window = MainWindow()
    window.output_nq_line_checkbox.setChecked(True)
    assert window.output_nq_value_spin.isEnabled() is True
    for canvas in window._all_canvases():
        assert canvas._show_nq_line is True
        assert canvas._nq_values == [1]

    window.output_nq_value_spin.setValue(3)
    for canvas in window._all_canvases():
        assert canvas._nq_values == [3]

    window.output_nq_line_checkbox.setChecked(False)
    assert window.output_nq_value_spin.isEnabled() is False
    for canvas in window._all_canvases():
        assert canvas._show_nq_line is False


def test_main_window_nq_line_draws_and_removes_a_line_on_a_real_2d_contour(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.process_indirect_button.setChecked(True)
    assert len(window.canvas.ax.lines) == 0

    window.output_nq_line_checkbox.setChecked(True)
    assert len(_trace_lines(window.canvas.ax)) == 1

    window.output_nq_line_checkbox.setChecked(False)
    assert len(window.canvas.ax.lines) == 0


def test_main_window_output_fontsize_and_ticks_apply(qapp):
    window = MainWindow()
    window.output_fontsize_spin.setValue(22)
    window.output_tick_length_spin.setValue(9.0)
    window.output_tick_width_spin.setValue(2.0)
    assert window.canvas._AXIS_LABELSIZE == 22
    assert window.canvas._TICK_LENGTH == pytest.approx(9.0)
    assert window.canvas._TICK_WIDTH == pytest.approx(2.0)


def test_main_window_output_legend_checkbox_controls_spectrum_canvas(qapp):
    window = MainWindow()
    window.output_legend_checkbox.setChecked(True)
    assert window.canvas._legend_visible is True
    window.output_legend_checkbox.setChecked(False)
    assert window.canvas._legend_visible is False


def test_main_window_output_legend_position_applies_to_spectrum_canvas(qapp):
    window = MainWindow()
    window.output_legend_loc_combo.setCurrentText("lower left")
    assert window.canvas._legend_loc == "lower left"


def test_main_window_compare_colormap_applies_to_the_current_plot(qapp):
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    window.output_colormap_combo.setCurrentText("plasma")
    assert window.compare_canvas._compare_colormap == "plasma"

    window._on_apply_style_to_all_clicked()
    for canvas in window._all_canvases():
        assert canvas._compare_colormap == "plasma"


def test_main_window_output_export_settings_stored_not_applied_live(qapp):
    # Export DPI/figure size are stored for File > Save Figure As...
    # only -- must not touch the live canvas figure at all (see
    # MplCanvas.set_export_settings's own docstring for why).
    window = MainWindow()
    original_dpi = window.canvas.figure.get_dpi()
    original_size = window.canvas.figure.get_size_inches()

    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.output_dpi_spin.setValue(600)
    window.output_fig_width_spin.setValue(10.0)
    window.output_fig_height_spin.setValue(7.0)

    assert window.canvas.figure.get_dpi() == pytest.approx(original_dpi)
    assert window.canvas.figure.get_size_inches() == pytest.approx(original_size)
    # Per plot now, like every other style setting -- Apply to all is
    # what puts them everywhere.
    assert window.canvas._export_dpi == pytest.approx(600)
    assert window.canvas._export_fig_width == pytest.approx(10.0)
    assert window.canvas._export_fig_height == pytest.approx(7.0)

    window._on_apply_style_to_all_clicked()
    for canvas in window._all_canvases():
        assert canvas._export_dpi == pytest.approx(600)
        assert canvas._export_fig_width == pytest.approx(10.0)
        assert canvas._export_fig_height == pytest.approx(7.0)


def test_main_window_save_and_load_figure_style_template(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window = MainWindow()
    window.output_linewidth_spin.setValue(2.5)
    window.output_fontsize_spin.setValue(18)
    window.output_type_combo.setCurrentIndex(window.output_type_combo.findData("compare"))

    window._on_save_figure_style_template()
    assert os.path.exists(window._FIGURE_STYLES_PATH)

    # A fresh window (defaults) loading the saved "compare" template
    # should pick the saved values back up.
    window2 = MainWindow()
    window2.output_type_combo.setCurrentIndex(window2.output_type_combo.findData("compare"))
    window2._on_load_figure_style_template()

    assert window2.output_linewidth_spin.value() == pytest.approx(2.5)
    assert window2.output_fontsize_spin.value() == 18
    assert window2.canvas._AXIS_LINEWIDTH == pytest.approx(2.5)


def test_main_window_load_figure_style_template_missing_shows_message(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    window = MainWindow()
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_load_figure_style_template()

    assert shown.get("called") is True


# ============================================================
# Per-probe app-wide config (imaging max gradient)
# ============================================================

def test_main_window_save_probe_gradient_default_writes_config_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(tmp_path / "probe_config.json"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="probecfg1d", probe="5 mm BBO"), vendor="bruker")

    window.imaging_max_gradient_spin.setValue(42.5)
    window._on_save_probe_gradient_default_clicked()

    config = window._load_probe_config()
    # Per gradient axis -- the three coils of a triple-axis probe are
    # rarely the same strength (see MainWindow._probe_axis_gradients).
    assert config == {"5 mm BBO": {"max_gradient_g_per_cm": {"z": pytest.approx(42.5)}}}


def test_main_window_save_probe_gradient_default_without_a_probe_shows_message(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(tmp_path / "probe_config.json"))
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="noprobe1d"), vendor="bruker")  # no probe kwarg
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_save_probe_gradient_default_clicked()

    assert shown.get("called") is True
    assert window._load_probe_config() == {}


def test_main_window_probe_default_gradient_applies_automatically_to_a_new_spectrum(qapp, tmp_path, monkeypatch):
    # Explicit user request: a saved probe default should be
    # "associated automatically with the probe" -- no separate load
    # step, unlike the figure-style templates above.
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(tmp_path / "probe_config.json"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="probesave1d", probe="5 mm BBO"), vendor="bruker")
    window.imaging_max_gradient_spin.setValue(42.5)
    window._on_save_probe_gradient_default_clicked()

    # A second, otherwise-unrelated spectrum recording the *same* probe.
    window._load(
        _make_raw_1d_dataset(tmp_path, monkeypatch, name="probeauto1d", probe="5 mm BBO"), vendor="bruker",
    )

    assert window.imaging_max_gradient_spin.value() == pytest.approx(42.5)


def test_main_window_per_experiment_gradient_overrides_the_probe_default(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(tmp_path / "probe_config.json"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="probesave1d", probe="5 mm BBO"), vendor="bruker")
    window.imaging_max_gradient_spin.setValue(42.5)
    window._on_save_probe_gradient_default_clicked()

    window._load(
        _make_raw_1d_dataset(tmp_path, monkeypatch, name="probeoverride1d", probe="5 mm BBO"), vendor="bruker",
    )
    index = window._selected_index()
    window.imaging_max_gradient_spin.setValue(10.0)  # this experiment's own explicit value

    window._select_single_row(0)
    window._select_single_row(index)  # back to the overridden one

    assert window.imaging_max_gradient_spin.value() == pytest.approx(10.0)  # not silently reverted to 42.5


def test_main_window_probe_default_gradient_does_not_apply_to_a_different_probe(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(tmp_path / "probe_config.json"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="probesave1d", probe="5 mm BBO"), vendor="bruker")
    window.imaging_max_gradient_spin.setValue(42.5)
    window._on_save_probe_gradient_default_clicked()

    window._load(
        _make_raw_1d_dataset(tmp_path, monkeypatch, name="otherprobe1d", probe="1.3 mm MAS"), vendor="bruker",
    )

    assert window.imaging_max_gradient_spin.value() == pytest.approx(0.0)


# ============================================================
# Imaging Analysis sub-tab (Phase 5: gradient-to-mm axis)
# ============================================================

def test_main_window_imaging_gradient_enables_mm_axis(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")  # NUC1=1H

    window.imaging_max_gradient_spin.setValue(50.0)

    assert window.canvas._imaging_gamma_hz_per_t == pytest.approx(42.577e6, rel=1e-3)
    assert window.canvas._imaging_max_gradient_g_per_cm == pytest.approx(50.0)
    assert "Hz/mm" in window.imaging_gradient_status_label.text()


def test_main_window_imaging_gradient_zero_disables_mm_axis(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    window.imaging_max_gradient_spin.setValue(50.0)
    window.imaging_max_gradient_spin.setValue(0.0)

    assert window.canvas._imaging_max_gradient_g_per_cm is None
    assert "Set a max gradient" in window.imaging_gradient_status_label.text()


def test_main_window_imaging_gradient_change_refreshes_active_mm_axis(qapp, tmp_path):
    # Regression test: changing the gradient while the Spectrum
    # canvas is already showing the mm axis used to leave it frozen
    # at whatever conversion was in effect when "mm" was first
    # selected -- looked like the plot "wasn't refreshing at all"
    # (and, by extension, that scroll-zoom on it was broken, though
    # zoom itself was never actually the problem).
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    window.imaging_max_gradient_spin.setValue(50.0)
    window.canvas.set_axis_unit("mm")
    x_at_50 = window.canvas.ax.lines[0].get_xdata().copy()

    window.imaging_max_gradient_spin.setValue(100.0)

    assert window.canvas.axis_unit == "mm"
    x_at_100 = window.canvas.ax.lines[0].get_xdata()
    assert not np.allclose(x_at_50, x_at_100)
    # Doubling the gradient should exactly halve the mm range.
    np.testing.assert_allclose(x_at_100, x_at_50 / 2.0, atol=1e-9)


def test_main_window_imaging_gradient_cleared_while_mm_active_falls_back_to_ppm(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    window.imaging_max_gradient_spin.setValue(50.0)
    window.canvas.set_axis_unit("mm")

    window.imaging_max_gradient_spin.setValue(0.0)

    assert window.canvas.axis_unit == "ppm"


def test_main_window_imaging_gradient_persists_per_spectrum(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window.imaging_max_gradient_spin.setValue(50.0)
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")

    assert window.imaging_max_gradient_spin.value() == pytest.approx(0.0)  # spectrum b has none set

    window._select_single_row(0)
    assert window.imaging_max_gradient_spin.value() == pytest.approx(50.0)


def test_main_window_imaging_gradient_recomputes_gamma_on_reselect(qapp, tmp_path):
    # A stale gamma from a *different* nucleus's spectrum must not
    # leak into the mm axis for whatever's newly selected.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="proton"), vendor="bruker")  # 1H
    window.imaging_max_gradient_spin.setValue(50.0)
    proton_gamma = window.canvas._imaging_gamma_hz_per_t

    carbon_path = _make_1d_dataset(tmp_path, name="carbon")
    # _make_1d_dataset always writes NUC1=1H -- overwrite with a
    # different nucleus for this test's own purpose.
    _write_params(
        tmp_path / "carbon" / "acqus",
        {"NUC1": "13C", "BF1": 100.0, "SFO1": 100.03, "O1": 2000.0, "NS": 4},
    )
    window._load(carbon_path, vendor="bruker")
    window.imaging_max_gradient_spin.setValue(50.0)

    assert window.canvas._imaging_gamma_hz_per_t != pytest.approx(proton_gamma)
    window._select_single_row(0)
    assert window.canvas._imaging_gamma_hz_per_t == pytest.approx(proton_gamma)


# ============================================================
# Compare deferred rendering (a pseudo-2D with many slices must not
# eagerly render a huge "Stacked" Compare view nobody is looking at)
# ============================================================

def test_main_window_compare_auto_populate_defers_render_when_not_visible(qapp, monkeypatch):
    # Regression test: loading a pseudo-2D series with hundreds of
    # slices used to eagerly render a full "Stacked" Compare view
    # (one subplot per slice) as a side effect of auto-populating the
    # list, even though nobody asked to see Compare -- measured at
    # ~25s for a real 1024-slice dataset. show_compare must not be
    # called at all here; only compare_list itself gets populated.
    delays = np.geomspace(0.001, 0.3, 50)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    called = []
    monkeypatch.setattr(window.compare_canvas, "show_compare", lambda *a, **k: called.append(True))

    _load_spectrum_directly(window, spectrum)  # auto-populates compare_list from its own slices

    assert window.compare_list.count() == 50
    assert called == []
    assert window._compare_view_stale is True


def test_main_window_compare_view_renders_once_tab_is_visited(qapp):
    delays = np.geomspace(0.001, 0.3, 50)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window._compare_view_stale is True

    window.canvas_tabs.setCurrentWidget(window.compare_canvas)

    assert window._compare_view_stale is False
    assert len(window.compare_canvas._compare_axes or []) == 50


def test_main_window_compare_explicit_plot_still_renders_immediately(qapp):
    # The user-initiated "Plot" button (switch_tab=True) must render
    # right away regardless of which tab is currently showing -- only
    # the silent auto-populate-from-pseudo-2D path defers.
    delays = np.geomspace(0.001, 0.3, 10)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.canvas_tabs.setCurrentWidget(window.canvas)  # navigate away first

    window._refresh_compare_view()  # the Plot button's own handler, switch_tab=True default

    assert window._compare_view_stale is False
    assert window.canvas_tabs.currentWidget() is window.compare_canvas


# ============================================================
# Sidecar-file persistence (io.persistence)
# ============================================================

def test_main_window_fid_processing_change_writes_sidecar_files(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="sidecar1d")
    window._load(path, vendor="bruker")

    window._set_zero_fill_value(128)
    window._on_fid_processing_control_changed()
    _settle_sidecar_save(window)

    from opennmr.io import persistence
    processing = persistence.load_processing_sidecar(path)
    assert processing["live_zero_fill"] == 128
    metadata = persistence.load_metadata_sidecar(path)
    assert metadata["vendor"] == "bruker"


def test_main_window_reopening_restores_zero_fill_and_recomputes_data(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="sidecarreopen1d")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window._set_zero_fill_value(128)
    window._on_fid_processing_control_changed()
    _settle_sidecar_save(window)
    assert len(window.session.spectra[0].data) == 128

    # A fresh MainWindow/session -- same on-disk data, nothing carried
    # over in memory -- proves this survives an app restart, not just
    # a reselection within one running session.
    window2 = MainWindow()
    window2._load(path, vendor="bruker")
    index = window2._selected_index()

    assert window2.zero_fill_custom_spin.value() == 128
    assert len(window2.session.spectra[index].data) == 128


def test_main_window_reopening_restores_phase(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="sidecarphase1d")
    window = MainWindow()
    window._load(path, vendor="bruker")
    original = window.session.spectra[0].data.copy()

    window.phase0_handle.setValue(45.0)
    window._capture_phase_reference()  # commits the pending phase delta into metadata.raw, same as reselecting
    _settle_sidecar_save(window)

    from opennmr.io import persistence
    saved = persistence.load_processing_sidecar(path)
    assert saved["phase0"] == pytest.approx(45.0)

    window2 = MainWindow()
    window2._load(path, vendor="bruker")
    restored = window2.session.spectra[0].data
    assert not np.allclose(original, restored)  # actually re-phased, not just bookkeeping


def test_main_window_reopening_restores_mass_and_nucleus_override(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="sidecarmeta1d")
    window = MainWindow()
    window._load(path, vendor="bruker")
    index = window._selected_index()

    window.session.set_mass(index, 0.045)
    window._schedule_sidecar_save(index)
    window._set_nucleus(index, "23Na")
    _settle_sidecar_save(window)

    window2 = MainWindow()
    window2._load(path, vendor="bruker")
    restored = window2.session.spectra[window2._selected_index()]
    assert restored.metadata.mass == pytest.approx(0.045)
    assert restored.metadata.nucleus == "23Na"


def test_main_window_reopening_a_2d_restores_f1_truncate(qapp, tmp_path, monkeypatch):
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="sidecar2d", f1_nuc="13C")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)  # opt into genuine 2D -- see F1-truncate tests' own comment
    window.tdeff_f1_spin.setValue(3)
    window.zero_fill_f1_spin.setValue(3)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)
    _settle_sidecar_save(window)
    assert window.session.spectra[0].data.shape[0] == 3

    window2 = MainWindow()
    window2._load(path, vendor="bruker")
    index = window2._selected_index()
    assert window2.session.spectra[index].data.shape[0] == 3
    assert window2.tdeff_f1_spin.value() == 3
    assert window2.zero_fill_f1_spin.value() == 3


def test_main_window_no_sidecar_files_is_a_silent_noop_on_load(qapp, tmp_path, monkeypatch):
    # The common, first-ever-load case -- nothing saved yet -- must
    # load exactly as it always did, no crash, no spurious restore.
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="nosidecar1d")
    window = MainWindow()
    window._load(path, vendor="bruker")
    assert len(window.session.spectra[0].data) == 64  # plain vendor-default length, untouched


def test_main_window_info_label_shows_probe_when_recorded(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="probeinfo1d", probe="5 mm BBO")
    window = MainWindow()
    window._load(path, vendor="bruker")

    assert "Probe: 5 mm BBO" in window.info_label.text()


def test_main_window_info_label_shows_pulse_programme_when_recorded(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="pulprog1d", pulprog="zg30")
    window = MainWindow()
    window._load(path, vendor="bruker")

    assert "Pulse programme: zg30" in window.info_label.text()


def test_main_window_info_label_notes_when_loaded_from_processed_data(qapp, tmp_path, monkeypatch):
    # Real user request: "note when the processed data is being read in
    # instead of raw data" -- io.bruker._read_auto tags
    # metadata.raw["loaded_from_processed"] with a short reason for
    # every fallback path (see that function's own docstring); the
    # info panel should surface it persistently (not just the existing
    # one-time popup at load time) so it's visible any time this
    # spectrum is reselected later, not only right after loading.
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="procnote1d")
    window = MainWindow()
    window._load(path, vendor="bruker")
    index = window._selected_index()
    spectrum = window.session.spectra[index]
    assert "loaded_from_processed" not in spectrum.metadata.raw  # sanity: not set by default

    spectrum.metadata.raw["loaded_from_processed"] = "No raw ser file found for this experiment."
    window._refresh_selection_display(index)

    assert "Loaded from processed data, not the raw FID" in window.info_label.text()
    assert "No raw ser file found for this experiment." in window.info_label.text()


def test_main_window_reselection_flushes_pending_sidecar_save_for_outgoing_spectrum(qapp, tmp_path, monkeypatch):
    # _schedule_sidecar_save only tracks one pending index at a time --
    # switching to spectrum B right after editing spectrum A (before A's
    # own debounce timer fires) would otherwise silently drop A's edit
    # instead of ever writing it to disk.
    path_a = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="flusha")
    path_b = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="flushb")
    window = MainWindow()
    window._load(path_a, vendor="bruker")
    window._load(path_b, vendor="bruker")

    window._select_single_row(0)
    window._set_zero_fill_value(128)
    window._on_fid_processing_control_changed()
    assert window._sidecar_save_timer.isActive()  # scheduled, not yet flushed

    window._select_single_row(1)  # switch away before the debounce interval elapses

    from opennmr.io import persistence
    assert persistence.load_processing_sidecar(path_a)["live_zero_fill"] == 128


def test_main_window_reset_processing_updates_the_saved_sidecar(qapp, tmp_path, monkeypatch):
    # Without this, an old saved sidecar would silently resurrect
    # exactly what Reset just discarded, the next time this experiment
    # is reopened.
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="resetsidecar1d")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window._set_zero_fill_value(128)
    window._on_fid_processing_control_changed()
    _settle_sidecar_save(window)

    from opennmr.io import persistence
    assert persistence.load_processing_sidecar(path)["live_zero_fill"] == 128

    window._on_reset_processing_clicked()
    _settle_sidecar_save(window)

    saved = persistence.load_processing_sidecar(path) or {}
    assert saved.get("live_zero_fill") != 128

    window2 = MainWindow()
    window2._load(path, vendor="bruker")
    assert len(window2.session.spectra[0].data) == 64  # plain vendor-default length, not the discarded 128


def test_main_window_select_single_row_draws_exactly_once(qapp, tmp_path, monkeypatch):
    # Regression test for a traced duplication: this widget has *two*
    # signals connected to handlers that both end in a full redraw
    # (currentRowChanged -> _on_selection_changed, and
    # itemSelectionChanged -> _on_list_selection_changed, which itself
    # calls _on_selection_changed), and setCurrentRow fires both -- so
    # every programmatic selection used to draw the whole spectrum
    # twice. On the real EXSY benchmark each redraw is ~4.2s, i.e. ~24%
    # of the load spent rendering a second identical picture.
    path = _make_1d_dataset(tmp_path)

    window = MainWindow()
    window._load(path, vendor="bruker")

    draws = []
    original = window._draw_selected_spectrum
    monkeypatch.setattr(
        window, "_draw_selected_spectrum",
        lambda *a, **k: (draws.append(1), original(*a, **k))[1],
    )

    window._select_single_row(0)

    assert len(draws) == 1, f"expected exactly one redraw, got {len(draws)}"


def test_main_window_select_single_row_still_selects_exactly_that_row(qapp, tmp_path):
    # The duplicate-redraw fix blocks the list's signals around the
    # three mutations -- so the selection state it produces still has
    # to be right, not just fast (blockSignals suppresses notification,
    # not the mutation itself, but that is exactly the kind of thing
    # worth pinning rather than assuming).
    window = MainWindow()
    for i in range(3):
        window._load(_make_1d_dataset(tmp_path, name=f"expt{i}"), vendor="bruker")

    window._select_single_row(1)

    assert window.spectrum_list.currentRow() == 1
    assert window._selected_rows() == [1]


def test_main_window_load_reads_a_genuine_2d_only_once(qapp, tmp_path, monkeypatch):
    # Regression test for the second traced duplication: _load used to
    # call session.add (a full genuine-2D read, F1 FFT and all) and
    # then session.reload(pseudo_2d=...), a complete *second* read from
    # disk, discarding the first result entirely. Measured at 5.19s of
    # pure waste per load on the real EXSY benchmark.
    path = _make_2d_dataset(tmp_path)

    reads = []
    original = opennmr_io.load
    monkeypatch.setattr(
        opennmr_io, "load",
        lambda *a, **k: (reads.append((a, k)), original(*a, **k))[1],
    )

    window = MainWindow()
    window._load(path, vendor="bruker")

    assert len(reads) == 1, f"expected a single read, got {len(reads)}"
    # ...and the one read it does perform must be the one that asks for
    # the treatment actually wanted, rather than reading the default and
    # correcting afterwards.
    assert reads[0][1].get("pseudo_2d") is True
    assert window.session.spectra[0].is_pseudo_2d is True


def test_main_window_load_respects_a_saved_genuine_2d_preference_in_one_read(qapp, tmp_path, monkeypatch):
    # The pre-read decision must honour a saved "Process Indirect
    # Dimension" opt-in (pseudo_2d: False) -- otherwise the fast path
    # would silently force pseudo-2D on datasets the user explicitly
    # opted out of, which is a correctness regression, not a perf one.
    path = _make_2d_dataset(tmp_path)
    # Written directly rather than via save_metadata_sidecar (which
    # takes a whole spectrum) -- this test only needs the one field the
    # pre-read decision actually consults.
    with open(io_persistence.metadata_sidecar_path(path), "w") as f:
        json.dump({"pseudo_2d": False}, f)

    reads = []
    original = opennmr_io.load
    monkeypatch.setattr(
        opennmr_io, "load",
        lambda *a, **k: (reads.append((a, k)), original(*a, **k))[1],
    )

    window = MainWindow()
    window._load(path, vendor="bruker")

    assert len(reads) == 1
    assert reads[0][1].get("pseudo_2d") is False


def test_read_auto_drops_pseudo_2d_for_one_dimensional_data(qapp, tmp_path):
    # _load can't know whether a path is 1D or 2D without doing the very
    # read it wants to parameterise, so it passes pseudo_2d in
    # unconditionally for Bruker; _read_auto is what makes that safe by
    # dropping it on the branches whose readers don't accept it. Without
    # this, every 1D Bruker load would raise TypeError.
    path = _make_1d_dataset(tmp_path)

    spectrum = io_bruker._read_auto(path, pseudo_2d=True)

    assert spectrum.ndim == 1


def test_main_window_magnitude_button_reflects_recorded_mode(qapp, tmp_path, monkeypatch):
    # A dataset whose own PH_mod asks for magnitude should come up with
    # the button already on, without the user doing anything.
    path = _make_1d_dataset(tmp_path)
    window = MainWindow()
    window._load(path, vendor="bruker")
    assert window.magnitude_button.isChecked() is False  # this fixture records no PH_mod

    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["magnitude_mode"] = "magnitude"
    window._refresh_selection_display(0)
    assert window.magnitude_button.isChecked() is True


def test_main_window_magnitude_toggle_reloads_with_explicit_flag(qapp, tmp_path, monkeypatch):
    # Magnitude is applied by the reader, deliberately with no second
    # in-memory implementation -- so the toggle must go through a
    # reload carrying an explicit `magnitude=`, not recompute locally.
    path = _make_1d_dataset(tmp_path)
    window = MainWindow()
    window._load(path, vendor="bruker")

    seen = []
    original = Session.reload
    monkeypatch.setattr(
        Session, "reload",
        lambda self, index, **kw: (seen.append(kw), original(self, index, **kw))[1],
    )

    window.magnitude_button.setChecked(True)

    assert seen and seen[-1].get("magnitude") is True
    # The explicit choice is recorded separately from the reader's own
    # descriptive flag, so "off" can be told apart from "never set".
    assert window.session.spectra[0].metadata.raw.get("magnitude_override") is True


def test_main_window_magnitude_off_is_remembered_for_a_file_that_records_it(qapp, tmp_path):
    # The case the separate override flag exists for: a file whose own
    # PH_mod asks for magnitude, which the user switched off. Without
    # an explicit saved False, the next load re-detects PH_mod and
    # silently turns it back on.
    path = _make_1d_dataset(tmp_path)
    # The processing sidecar wraps its payload under "state" -- written
    # in that shape rather than as a bare dict, which load_processing_
    # sidecar silently ignores.
    with open(io_persistence.processing_sidecar_path(path), "w") as f:
        json.dump({"state": {"magnitude_override": False}}, f)

    seen = {}
    window = MainWindow()
    real_add = window.session.add
    window.session.add = lambda p, vendor=None, **kw: (seen.update(kw), real_add(p, vendor=vendor, **kw))[1]
    window._load(path, vendor="bruker")

    assert seen.get("magnitude") is False


def test_main_window_magnitude_button_hidden_for_non_bruker(qapp, tmp_path):
    # Bruker is the only reader with a `magnitude` parameter today --
    # showing the button elsewhere would offer a control that raises.
    window = MainWindow()
    spectrum = _make_1d_peak_spectrum()
    spectrum.metadata.vendor = "xpulse"
    _load_spectrum_directly(window, spectrum)
    assert window.magnitude_button.isVisible() is False


def test_main_window_zero_phase_zeros_the_applied_phase_not_just_the_delta(qapp, tmp_path, monkeypatch):
    # Real user report: "Zeroing phases should actually zero the values,
    # not just reset to the loaded values." The spin box holds only the
    # delta since the last phase-reference capture; the total in force
    # is that delta plus whatever is committed in metadata.raw -- which
    # on a freshly loaded Bruker spectrum is the file's own recorded
    # PHC0/PHC1. Zeroing the delta therefore returned to the *loaded*
    # phase, visibly doing nothing on a spectrum not yet dragged.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["phase0"] = 120.0
    spectrum.metadata.raw["phase1"] = 45.0

    window._on_zero_phase_clicked(0)
    # The *applied* phase is what must reach zero -- asserted through
    # _phase_committed_value, which sums the file's own recorded phase
    # and the user's, since those are stored separately and use
    # different models.
    assert window._phase_committed_value(0) == pytest.approx(0.0)
    assert window.phase0_spin.value() == pytest.approx(0.0)
    assert window._phase_committed_value(1) == pytest.approx(45.0)  # untouched

    window._on_zero_phase_clicked(1)
    assert window._phase_committed_value(1) == pytest.approx(0.0)


def test_main_window_reset_phase_zeros_both_applied_phases(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["phase0"] = -30.0
    spectrum.metadata.raw["phase1"] = 200.0

    window._on_reset_phase_clicked()

    assert window._phase_committed_value(0) == pytest.approx(0.0)
    assert window._phase_committed_value(1) == pytest.approx(0.0)


def test_main_window_zero_phase_with_no_committed_phase_is_still_zero(qapp, tmp_path, monkeypatch):
    # Nothing committed -> the delta that zeros the total is itself 0,
    # i.e. the old behaviour, so this must not regress the common case.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.session.spectra[0].metadata.raw["phase0"] = 0.0
    window.session.spectra[0].metadata.raw["recorded_phase0"] = 0.0
    window.phase0_spin.setValue(37.0)

    window._on_zero_phase_clicked(0)

    assert window.phase0_spin.value() == pytest.approx(0.0)
    assert window._phase_committed_value(0) == pytest.approx(0.0)


@pytest.mark.parametrize("dataset,genuine_2d", [
    ("bruker_T2/11", False),
    ("bruker_PFG/150", False),
    ("Bruker Example Data/Cyclosporine/203", True),      # Echo-Antiecho
    ("Bruker Example Data/exam_CMCse_3/19", True),       # States-TPPI
    ("Bruker Example Data/exam_CMCse_3/12", True),       # magnitude mode
])
def test_main_window_live_rebuild_reproduces_the_load_exactly(qapp, dataset, genuine_2d, monkeypatch, tmp_path):
    # THE acceptance test for separating the recorded phase from the
    # user's. The GUI rebuilds from the cached raw FID, which is the FID
    # *before* the reader phased anything -- so every reconstruction
    # path has to re-apply the file's own recorded phase, in the file's
    # own model, or touching any control makes the spectrum jump.
    #
    # This is exactly what blocked shipping the phase-convention fix the
    # first time: measured at 0.46 (T2) and 0.76 (PFG) relative
    # difference before the split, and 1.93 for the magnitude-mode COSY
    # (whose rebuild path dropped magnitude entirely).
    path = os.path.join("test_data", *dataset.split("/"))
    if not os.path.isdir(path):
        pytest.skip(f"needs the real {dataset} dataset")

    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(path, vendor="bruker")
    if genuine_2d:
        window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    # Baseline correction off: it is a separate, deliberately deferred
    # step, not part of what this test is pinning.
    spectrum.metadata.raw["baseline_correct"] = False
    before = np.array(spectrum.data)

    window._recompute_multi_row_data(spectrum)

    after = np.array(window.session.spectra[0].data)
    scale = max(np.max(np.abs(before)), 1e-30)
    assert np.max(np.abs(after - before)) / scale < 1e-6


def test_main_window_recorded_and_user_phase_are_stored_separately(qapp, tmp_path, monkeypatch):
    # They are applied through different models, so merging them into
    # one number would make neither reproducible.
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64)
    window = MainWindow()
    window._load(path, vendor="bruker")
    raw = window.session.spectra[0].metadata.raw

    assert "recorded_phase0" in raw and "recorded_phase1" in raw
    assert raw["phase0"] == pytest.approx(0.0)  # the user has dialled nothing yet


def test_main_window_loading_does_not_force_a_navigation_mode(qapp, tmp_path, monkeypatch):
    # Pan used to be switched on for the first spectrum loaded (an
    # earlier explicit request) and was withdrawn by a later one:
    # "actually stop having pan as the default mode". A mode the
    # application turns on for you is a mode you have to notice and
    # turn off.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    assert window._navigation_mode() == ""

    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")

    assert window._navigation_mode() == ""
    # Still one click (or Ctrl+Shift+P) away, just not automatic.
    window._set_navigation_mode("pan")
    assert window._navigation_mode() == "pan"


def test_main_window_navigation_mode_toggle_and_idempotence(qapp, tmp_path, monkeypatch):
    # matplotlib's toolbar pan()/zoom() *toggle*, so setting a mode that
    # is already active would turn it back off -- the helper has to
    # check first. Pinned because it is an easy and invisible mistake.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window._set_navigation_mode("pan")  # no longer automatic on load

    window._toggle_navigation_mode()
    assert window._navigation_mode() == "zoom"
    window._toggle_navigation_mode()
    assert window._navigation_mode() == "pan"

    window._set_navigation_mode("zoom")
    window._set_navigation_mode("zoom")   # must not toggle back off
    assert window._navigation_mode() == "zoom"


def test_main_window_navigation_mode_applies_to_every_canvas(qapp, tmp_path, monkeypatch):
    # The mode is a per-toolbar flag, so leaving the other canvases
    # alone would mean switching canvas tab silently changed the
    # interaction mode back.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window._set_navigation_mode("zoom")

    modes = {
        canvas.toolbar.mode.name if hasattr(canvas.toolbar.mode, "name") else str(canvas.toolbar.mode)
        for canvas in window._all_canvases()
    }
    assert len(modes) == 1 and "ZOOM" in modes.pop().upper()


def test_main_window_keeping_an_extracted_slice_makes_it_phasable(qapp, tmp_path, monkeypatch):
    # Real user request: "a method to phase on an extracted slice".
    # extract_slice only *displayed* a trace -- it never reached the
    # session, and every phase control acts on session spectra.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    before = len(window.session)

    window.canvas.extract_slice("f1", 1)
    assert window.canvas._extracted_trace is not None
    window.canvas._keep_slice_button.click()

    assert len(window.session) == before + 1
    kept = window.session.spectra[-1]
    assert kept.ndim == 1
    # "derived": it has no raw FID to reprocess from, so the FID
    # controls stay hidden while the phase controls still work.
    assert kept.metadata.vendor == "derived"
    # Phasable, which is the whole point -- get_trace carries data_imag.
    assert kept.data_imag is not None
    assert window._selected_index() == len(window.session) - 1

    before_data = np.array(window.session.spectra[-1].data)
    window.phase0_spin.setValue(90.0)
    assert not np.allclose(before_data, window.session.spectra[-1].data)


def _keep_a_slice(window, axis="f1", index=1):
    """Extract a slice and keep it, returning the kept spectrum."""
    window.canvas.extract_slice(axis, index)
    window.canvas._keep_slice_button.click()
    return window.session.spectra[-1]


def test_main_window_kept_slice_records_its_source(qapp, tmp_path, monkeypatch):
    # The slice is a working copy of part of the 2D, not a dead end --
    # it keeps a live link back so its phase can be pushed home.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    parent = window.session.spectra[0]

    kept = _keep_a_slice(window)

    assert kept.metadata.raw["slice_parent"] is parent
    assert kept.metadata.raw["slice_axis"] == "f1"
    # Fixing F1 traces *along F2*, so the slice's own phase is an F2
    # phase. Getting this backwards would silently phase the wrong
    # dimension of the 2D.
    assert window._slice_source_of(kept) == (parent, "f2")


def test_main_window_slice_phase_applies_to_the_source_2d(qapp, tmp_path, monkeypatch):
    # Real user request: "I don't think the derived slice phasing is
    # feeding back into the contour... have some way to go back into
    # the contour data."
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    parent = window.session.spectra[0]
    parent.metadata.raw["baseline_correct"] = False
    before = np.array(parent.data)

    kept = _keep_a_slice(window)
    window.phase0_spin.setValue(60.0)
    window.apply_slice_phase_button.click()

    # The 2D picked the phase up, and the view returned to it.
    assert parent.metadata.raw["phase0"] == pytest.approx(60.0)
    assert window._selected_index() == 0
    assert not np.allclose(before, window.session.spectra[0].data)
    # Cleared on the slice, so a second click can't double-apply.
    assert kept.metadata.raw["phase0"] == pytest.approx(0.0)
    assert window.phase0_spin.value() == pytest.approx(0.0)


def test_main_window_slice_phase_matches_phasing_the_2d_directly(qapp, tmp_path, monkeypatch):
    # The whole point: routing a phase through a slice must land in
    # exactly the same place as dialling it on the 2D itself, because
    # both go through _recompute_multi_row_data rather than one of them
    # mutating data on its own.
    _isolate_sidecars(monkeypatch, tmp_path)

    def _settle(window):
        if window._live_reprocess_timer.isActive():
            window._live_reprocess_timer.stop()
            window._run_debounced_multi_row_reprocess()

    via_slice = MainWindow()
    via_slice._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, name="viaslice"), vendor="bruker")
    _keep_a_slice(via_slice)
    via_slice.phase0_spin.setValue(60.0)
    via_slice.apply_slice_phase_button.click()
    _settle(via_slice)

    direct = MainWindow()
    direct._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, name="direct"), vendor="bruker")
    direct.phase0_spin.setValue(60.0)
    _settle(direct)

    a = np.asarray(via_slice.session.spectra[0].data)
    b = np.asarray(direct.session.spectra[0].data)
    assert np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30) < 1e-9


def test_main_window_apply_slice_phase_button_only_shown_for_a_slice(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    assert window.apply_slice_phase_button.isHidden() is True

    _keep_a_slice(window)
    assert window.apply_slice_phase_button.isHidden() is False

    # Back on the 2D itself, it is meaningless again.
    window._select_single_row(0)
    assert window.apply_slice_phase_button.isHidden() is True


def test_main_window_slice_source_is_dropped_if_the_2d_is_removed(qapp, tmp_path, monkeypatch):
    # Holding a reference to a spectrum no longer in the session would
    # let the button apply a phase to something the user can't see.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    kept = _keep_a_slice(window)
    assert window._slice_source_of(kept)[0] is not None

    window.session.remove(0)
    assert window._slice_source_of(kept) == (None, None)


def test_main_window_live_phase_keeps_the_baseline_after_a_settle(qapp, tmp_path, monkeypatch):
    # Real user report: "The indirect phasing sort of bounces like there
    # is a mismatch to the current and read phases."
    #
    # Mechanism: baseline correction is deliberately NOT applied on
    # every drag tick (measured at 173 ms for this dataset and 1.5 s for
    # the EXSY -- far too slow), so it is deferred to a debounced
    # rebuild. That rebuild fires whenever the drag pauses for 200 ms.
    # Before this fix the *next* tick re-derived the display from the
    # pre-drag snapshot, throwing the baseline away again -- so the view
    # alternated between corrected and uncorrected on every pause.
    #
    # The fix makes the rebuild's own result the new reference, so the
    # drag continues from it. Asserted against the authoritative answer
    # (a clean rebuild at the same phase) rather than against a raw
    # difference, since a phase change legitimately alters the data.
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip("needs the real Cyclosporine/203 dataset")
    _isolate_sidecars(monkeypatch, tmp_path)

    def _build(phases, settle_at):
        window = MainWindow()
        window._load(path, vendor="bruker")
        window._on_process_indirect_toggled(True)
        spectrum = window.session.spectra[0]
        spectrum.metadata.raw["baseline_correct"] = True
        # F1. F2 on a hypercomplex 2D has no live tick to bounce any
        # more -- see the test below.
        window.dim_combo.setCurrentIndex(1)
        for value in phases:
            window.phase0_spin.setValue(value)
            if value in settle_at and window._live_reprocess_timer.isActive():
                window._live_reprocess_timer.stop()
                window._run_debounced_multi_row_reprocess()
        return np.asarray(spectrum.data)

    # Settle at 30, then one more tick to 30.5 -- what the user sees.
    live = _build([30.0, 30.5], {30.0})
    # The authoritative answer for 30.5.
    authoritative = _build([30.5], {30.5})

    scale = max(np.max(np.abs(authoritative)), 1e-30)
    assert np.max(np.abs(live - authoritative)) / scale < 1e-3


def test_main_window_pick_mode_shows_a_slice_for_a_pseudo_2d(qapp, tmp_path, monkeypatch):
    # Real user report: "peak picking on a pseudo 2D is currently
    # impossible". Manual picking listens on the Spectrum canvas, which
    # shows the contour/stack for a pseudo-2D -- so a click yielded a 2D
    # position with no single trace to attach a peak to. Turning pick
    # mode on now puts the chosen slice up on its own.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    spectrum = window.session.spectra[0]
    window._select_single_row(0)

    assert not window.peak_slice_row.isHidden()
    assert window.peak_slice_spin.maximum() == spectrum.data.shape[0]
    assert window.peak_slice_spin.value() == 1          # first by default

    assert window.canvas._extracted_from_spectrum is None
    window.pick_peak_button.setChecked(True)
    assert window.canvas._extracted_from_spectrum is not None
    assert window.canvas._extracted_index == 1

    # Changing the slice while picking follows along.
    window.peak_slice_spin.setValue(4)
    assert window.canvas._extracted_index == 4


def test_main_window_peak_slice_spin_wraps_at_both_ends(qapp, tmp_path, monkeypatch):
    # Explicit user request: "if there's a box with a number, pressing
    # down on the lowest should cycle back to the highest".
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._select_single_row(0)
    n_rows = window.session.spectra[0].data.shape[0]

    assert window.peak_slice_spin.wrapping()
    window.peak_slice_spin.setValue(1)
    window.peak_slice_spin.stepBy(-1)
    assert window.peak_slice_spin.value() == n_rows
    window.peak_slice_spin.stepBy(1)
    assert window.peak_slice_spin.value() == 1


def test_main_window_peak_slice_row_hidden_for_a_1d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._select_single_row(0)
    assert window.peak_slice_row.isHidden()


def test_main_window_auto_pick_uses_the_selected_slice(qapp, tmp_path, monkeypatch):
    # Auto-Pick was hard-wired to slice 1 with no way to choose another,
    # and refused a conventional 2D outright.
    _isolate_sidecars(monkeypatch, tmp_path)
    # Rows must differ from each other, or "picks on slice 3" and
    # "picks on slice 1" are indistinguishable -- the default fixture
    # tiles one identical row, which made an earlier version of this
    # test pass even with the hard-wired index=1 still in place.
    n_f1, n_f2 = 8, 32
    t2 = np.arange(n_f2) / 8000.0
    fid = np.stack([
        (k + 1) * np.exp(-t2 * 300) * np.exp(2j * np.pi * (500.0 + 200.0 * k) * t2)
        for k in range(n_f1)
    ]).astype(np.complex128)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=n_f1, n_f2=n_f2, fid=fid),
                 vendor="bruker")
    spectrum = window.session.spectra[0]
    window._select_single_row(0)

    window.peak_slice_spin.setValue(3)
    target = window._peak_pick_target(spectrum)
    expected = processing_slicing.get_trace(spectrum, axis="f1", index=3)
    np.testing.assert_allclose(target.data, expected.data)
    slice_one = processing_slicing.get_trace(spectrum, axis="f1", index=1)
    assert not np.allclose(target.data, slice_one.data)

    # A genuine 2D is no longer refused outright either -- it picks on
    # one row at a time, exactly like a pseudo-2D.
    window._on_process_indirect_toggled(True)
    genuine = window.session.spectra[0]
    window._refresh_peak_slice_control(genuine)
    assert not window.peak_slice_row.isHidden()
    assert window._peak_pick_target(genuine).ndim == 1


def test_main_window_contour_slices_are_stored_against_the_spectrum(qapp, tmp_path, monkeypatch):
    # A slice at F1 index 412 means nothing on any other dataset, so
    # like the nQ line it belongs to the spectrum, not the canvas.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, name="one"), vendor="bruker")
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, name="two"), vendor="bruker")

    window._select_single_row(0)
    window.canvas._add_contour_slice("f1", 2, "Impurity")
    window.canvas._add_contour_slice("f1", 5, "Shift")
    stored = window.session.spectra[0].metadata.raw["contour_slices"]
    assert [(s["axis"], s["index"], s["label"]) for s in stored] == [
        ("f1", 2, "Impurity"), ("f1", 5, "Shift"),
    ]
    # Each carries its own colour, stored rather than re-derived at
    # draw time -- that is what keeps it across a unit change.
    assert all(s["color"] for s in stored)
    assert stored[0]["color"] != stored[1]["color"]

    window._select_single_row(1)
    assert window.canvas._contour_slices == []
    assert "contour_slices" not in window.session.spectra[1].metadata.raw

    window._select_single_row(0)
    assert len(window.canvas._contour_slices) == 2

    window.canvas._clear_contour_slices()
    assert window.session.spectra[0].metadata.raw["contour_slices"] == []


def test_main_window_restoring_contour_slices_does_not_write_back(qapp, tmp_path, monkeypatch):
    # set_contour_slices deliberately doesn't emit -- otherwise pushing
    # a spectrum's stored slices onto the canvas would immediately write
    # the same list straight back out as if the user had just added it.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    spectrum = window.session.spectra[0]

    emitted = []
    window.canvas.contourSlicesChanged.connect(emitted.append)
    window._restore_contour_slices(spectrum)
    assert emitted == []
    assert "contour_slices" not in spectrum.metadata.raw


def test_main_window_nq_line_is_per_spectrum_not_app_wide(qapp, tmp_path, monkeypatch):
    # User request: "plotting options like nQ line should be tied to the
    # loaded spectrum." A diagonal means something on a homonuclear
    # correlation and nothing on a heteronuclear one, and n is a
    # property of the experiment -- as one global switch it followed the
    # user onto every other spectrum, including the ones it was
    # meaningless on.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, name="first"), vendor="bruker")
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, name="second"), vendor="bruker")

    window._select_single_row(0)
    window.output_nq_line_checkbox.setChecked(True)
    window.output_nq_value_spin.setValue(3)
    assert window.session.spectra[0].metadata.raw["nq_line"] is True
    assert window.session.spectra[0].metadata.raw["nq_value"] == 3

    # Selecting the other spectrum must not carry it across.
    window._select_single_row(1)
    assert not window.output_nq_line_checkbox.isChecked()
    assert "nq_line" not in window.session.spectra[1].metadata.raw

    # ...and coming back restores it, canvases included.
    window._select_single_row(0)
    assert window.output_nq_line_checkbox.isChecked()
    assert window.output_nq_value_spin.value() == 3
    assert window.output_nq_value_spin.isEnabled()


def test_main_window_nq_line_restore_does_not_write_back(qapp, tmp_path, monkeypatch):
    # Restoring the control from the spectrum must not look like the
    # user setting it -- otherwise selecting a spectrum that has never
    # had an nQ line would stamp a default onto it (and dirty its
    # sidecar) just by being looked at.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    spectrum = window.session.spectra[0]

    window._restore_nq_line_control(spectrum)
    assert "nq_line" not in spectrum.metadata.raw
    assert "nq_value" not in spectrum.metadata.raw


def test_main_window_nq_line_survives_a_reload(qapp, tmp_path, monkeypatch):
    # It rides along with the rest of the processing state, so it has to
    # come back on a fresh load of the same experiment.
    _isolate_sidecars(monkeypatch, tmp_path)
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8)
    window = MainWindow()
    window._load(path, vendor="bruker")
    window._select_single_row(0)
    window.output_nq_line_checkbox.setChecked(True)
    window.output_nq_value_spin.setValue(2)
    io_persistence.save_processing_sidecar(window.session.spectra[0])

    reopened = MainWindow()
    reopened._load(path, vendor="bruker")
    reopened._select_single_row(0)
    assert reopened.output_nq_line_checkbox.isChecked()
    assert reopened.output_nq_value_spin.value() == 2


def test_main_window_custom_equation_uses_acquisition_constants(qapp, tmp_path, monkeypatch):
    # "It would be good if I can make fitting equations using values
    # from the input data (e.g. delays and constants and L or
    # whatever)." Every acquisition parameter is offered rather than a
    # guessed subset -- build_custom_model substitutes only the names an
    # equation actually uses, so an unused one costs nothing.
    _isolate_sidecars(monkeypatch, tmp_path)
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    window = MainWindow()
    window._load(path, vendor="bruker")
    window._select_single_row(0)

    window.custom_equation_edit.setText("A*exp(-x*d1/T2)+C")
    window._on_custom_equation_changed(window.custom_equation_edit.text())
    label = window.custom_params_label.text()
    assert "A, T2, C" in label          # d1 is not among them
    assert "d1=2" in label              # ...and the preview says why

    spec = window._build_custom_relaxation_model()
    assert spec["params"] == ["A", "T2", "C"]
    assert spec["constants"] == {"d1": pytest.approx(2.0)}

    # A symbol nobody supplies is still fitted, exactly as before.
    window.custom_equation_edit.setText("A*exp(-x/Tq)+C")
    assert window._build_custom_relaxation_model()["params"] == ["A", "Tq", "C"]


def test_main_window_sr_box_shows_the_file_s_own_reference(qapp, tmp_path, monkeypatch):
    # User request: "I want any sr that was noted in the import files to
    # be applied already and to show in the processing tab." It was
    # always applied -- _ppm_hz_axis folds (SF - BF1) into every axis it
    # builds -- but nothing recorded how much, so the box read 0 for a
    # dataset that plainly was referenced.
    _isolate_sidecars(monkeypatch, tmp_path)
    path = os.path.join("test_data", "2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119", "2019")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    window = MainWindow()
    window._load(path, vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)

    assert spectrum.metadata.raw["sr_recorded_hz"] == pytest.approx(315.3)
    assert window.sr_shift_spin.value() == pytest.approx(315.3)
    assert window.sr_shift_indirect_spin.value() == pytest.approx(315.3, abs=1e-6)


def test_main_window_sr_box_is_the_total_not_a_delta(qapp, tmp_path, monkeypatch):
    # Showing the file's own value means the number in the box is the
    # *total* SR in effect, so _apply_reference_shift_hz applies
    # value - recorded to the as-loaded axis. Applying the unchanged
    # value must therefore be a no-op, and typing a different one means
    # "make the total this".
    _isolate_sidecars(monkeypatch, tmp_path)
    path = os.path.join("test_data", "2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119", "2019")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    window = MainWindow()
    window._load(path, vendor="bruker")
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)
    as_loaded = spectrum.axes[-1].ppm.copy()

    window._on_apply_sr_clicked()
    np.testing.assert_allclose(spectrum.axes[-1].ppm, as_loaded)

    window.sr_shift_spin.setValue(415.3)          # +100 Hz on the total
    window._on_apply_sr_clicked()
    np.testing.assert_allclose(spectrum.axes[-1].ppm,
                               as_loaded + 100.0 / spectrum.metadata.sfo1)

    window.sr_shift_spin.setValue(315.3)          # back to the file's own
    window._on_apply_sr_clicked()
    np.testing.assert_allclose(spectrum.axes[-1].ppm, as_loaded)


def test_main_window_imaging_gradient_uses_gpz1_as_a_percentage(qapp, tmp_path, monkeypatch):
    # Real user report: "the relative gradient strength doesn't seem to
    # be calculated correctly for my Bruker data (should be gpz1
    # generally as a percentage value, e.g. gpz1=2 is 2%)". GPZ1 was not
    # read at all -- the mm axis used the probe's full maximum, so at
    # gpz1=2 the scale came out 50x too small.
    _isolate_sidecars(monkeypatch, tmp_path)
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    window = MainWindow()
    window._load(path, vendor="bruker")
    window._select_single_row(0)
    spectrum = window.session.spectra[0]

    assert window._recorded_gradient_percent(spectrum) == pytest.approx(80.0)
    assert window.imaging_gradient_percent_spin.value() == pytest.approx(80.0)

    window.imaging_max_gradient_spin.setValue(50.0)
    assert window.canvas._imaging_max_gradient_g_per_cm == pytest.approx(40.0)

    # "generally" is not "always" -- the override wins and is stored.
    window.imaging_gradient_percent_spin.setValue(2.0)
    assert window.canvas._imaging_max_gradient_g_per_cm == pytest.approx(1.0)
    assert spectrum.metadata.raw["imaging_gradient_percent"] == pytest.approx(2.0)


def test_main_window_imaging_gradient_percent_defaults_to_full_without_gpz(qapp, tmp_path, monkeypatch):
    # No GPZ array (a non-Bruker dataset, or one whose parameters were
    # not retained) falls back to 100%, i.e. exactly the old behaviour.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._select_single_row(0)
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw.pop("acqus", None)

    assert window._recorded_gradient_percent(spectrum) is None
    window._restore_imaging_gradient_percent(spectrum)
    assert window.imaging_gradient_percent_spin.value() == pytest.approx(100.0)


def _answer_calibration_dialog(monkeypatch, values, accept=True):
    """Fill the "Set Reference from Cursor" dialog's ppm boxes and
    accept (or cancel) it, in creation order: direct, then indirect."""
    from PySide6.QtWidgets import QDialog, QDoubleSpinBox

    def fake_exec(self):
        if not accept:
            return QDialog.DialogCode.Rejected
        for spin, value in zip(self.findChildren(QDoubleSpinBox), values):
            spin.setValue(value)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)


def test_main_window_ppm_referencing_sets_both_dimensions(qapp, tmp_path, monkeypatch):
    # User request: "I need a right click option on the 2D spectra to be
    # able to set the sr (by setting ppm values)." Both dimensions at
    # once for a genuine 2D -- a heteronuclear pair needs referencing on
    # each.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)
    f2_before = spectrum.axes[1].ppm.copy()
    f1_before = spectrum.axes[0].ppm.copy()

    _answer_calibration_dialog(monkeypatch, [2.0, 20.0])
    window._on_reference_calibration_from_plot((2.5, 25.0))

    assert spectrum.metadata.raw["sr_shift_hz"] == pytest.approx(
        (2.0 - 2.5) * spectrum.metadata.sfo1)
    assert spectrum.metadata.raw["sr_shift_hz_f1"] == pytest.approx(
        (20.0 - 25.0) * spectrum.metadata.sfo1_indirect)
    # The axes really move, by exactly the requested correction.
    np.testing.assert_allclose(spectrum.axes[1].ppm, f2_before + (2.0 - 2.5))
    np.testing.assert_allclose(spectrum.axes[0].ppm, f1_before + (20.0 - 25.0))


def test_main_window_ppm_referencing_reads_the_current_axis_not_the_original(qapp, tmp_path, monkeypatch):
    # The clicked position comes off the axis *as displayed*, which
    # already carries whatever SR is applied -- so the new value is the
    # existing one plus the correction. _apply_reference_shift_hz is
    # idempotent against the original as-loaded axis, so treating the
    # correction as the whole value would be wrong twice over.
    # Asserted as: re-referencing a peak that already reads correctly
    # changes nothing.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)

    _answer_calibration_dialog(monkeypatch, [2.0, 20.0])
    window._on_reference_calibration_from_plot((2.5, 25.0))
    after_first = (spectrum.metadata.raw["sr_shift_hz"],
                   spectrum.metadata.raw["sr_shift_hz_f1"],
                   spectrum.axes[1].ppm.copy(), spectrum.axes[0].ppm.copy())

    # The peak now sits at 2.0 / 20.0; saying so again must be a no-op.
    _answer_calibration_dialog(monkeypatch, [2.0, 20.0])
    window._on_reference_calibration_from_plot((2.0, 20.0))
    assert spectrum.metadata.raw["sr_shift_hz"] == pytest.approx(after_first[0])
    assert spectrum.metadata.raw["sr_shift_hz_f1"] == pytest.approx(after_first[1])
    np.testing.assert_allclose(spectrum.axes[1].ppm, after_first[2])
    np.testing.assert_allclose(spectrum.axes[0].ppm, after_first[3])


def test_main_window_ppm_referencing_skips_a_pseudo_2d_indirect_axis(qapp, tmp_path, monkeypatch):
    # A pseudo-2D's indirect axis is a slice index, so there is nothing
    # to reference against there -- only the direct dimension is offered
    # or set.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)

    _answer_calibration_dialog(monkeypatch, [2.0, 20.0])
    window._on_reference_calibration_from_plot((2.5, 4.0))

    assert spectrum.metadata.raw["sr_shift_hz"] == pytest.approx(
        (2.0 - 2.5) * spectrum.metadata.sfo1)
    assert "sr_shift_hz_f1" not in spectrum.metadata.raw


def test_main_window_ppm_referencing_cancel_changes_nothing(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    before = spectrum.axes[1].ppm.copy()

    _answer_calibration_dialog(monkeypatch, [2.0, 20.0], accept=False)
    window._on_reference_calibration_from_plot((2.5, 25.0))

    assert "sr_shift_hz" not in spectrum.metadata.raw
    np.testing.assert_allclose(spectrum.axes[1].ppm, before)


def test_main_window_indirect_sr_shifts_only_the_indirect_axis(qapp, tmp_path, monkeypatch):
    # User request: "we need to add sr for indirect dimension in 2Ds."
    # The two dimensions need independent SR values against independent
    # base frequencies -- on a heteronuclear 2D they are different
    # nuclei, so one Hz offset cannot reference both.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)

    f1_before = spectrum.axes[0].ppm.copy()
    f2_before = spectrum.axes[1].ppm.copy()

    window.sr_shift_indirect_spin.setValue(100.0)
    window._on_apply_sr_clicked()
    np.testing.assert_allclose(
        spectrum.axes[0].ppm, f1_before + 100.0 / spectrum.metadata.sfo1_indirect,
    )
    np.testing.assert_allclose(spectrum.axes[1].ppm, f2_before)

    # Independent: a direct-dimension SR moves only F2, and does not
    # disturb the indirect shift already applied.
    window.sr_shift_spin.setValue(50.0)
    window._on_apply_sr_clicked()
    np.testing.assert_allclose(
        spectrum.axes[1].ppm, f2_before + 50.0 / spectrum.metadata.sfo1,
    )
    np.testing.assert_allclose(
        spectrum.axes[0].ppm, f1_before + 100.0 / spectrum.metadata.sfo1_indirect,
    )

    # A value, not a delta: re-applying is a no-op, and zero restores.
    window._on_apply_sr_clicked()
    np.testing.assert_allclose(
        spectrum.axes[0].ppm, f1_before + 100.0 / spectrum.metadata.sfo1_indirect,
    )
    window.sr_shift_indirect_spin.setValue(0.0)
    window.sr_shift_spin.setValue(0.0)
    window._on_apply_sr_clicked()
    np.testing.assert_allclose(spectrum.axes[0].ppm, f1_before)
    np.testing.assert_allclose(spectrum.axes[1].ppm, f2_before)


def test_main_window_apply_sr_reads_both_boxes_before_applying_either(qapp, tmp_path, monkeypatch):
    # _apply_reference_shift_hz ends by calling _restore_sr_control,
    # which rewrites *both* spin boxes from metadata. Reading the second
    # box after applying the first therefore handed back the stored
    # value (0) rather than what the user had just typed, and the
    # indirect shift silently did nothing. Found by running it.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)

    # Direct first in apply order, indirect second -- the one that got lost.
    window.sr_shift_spin.setValue(25.0)
    window.sr_shift_indirect_spin.setValue(75.0)
    window._on_apply_sr_clicked()

    assert spectrum.metadata.raw["sr_shift_hz"] == pytest.approx(25.0)
    assert spectrum.metadata.raw["sr_shift_hz_f1"] == pytest.approx(75.0)


def test_main_window_indirect_sr_row_hidden_without_an_indirect_frequency_axis(qapp, tmp_path, monkeypatch):
    # Nothing to reference against on a 1D, or on a pseudo-2D whose
    # indirect axis is a slice index rather than a frequency.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    spectrum = window.session.spectra[0]

    window._restore_sr_control(spectrum)          # pseudo-2D by default
    assert window.sr_shift_indirect_spin.isHidden()
    assert window.sr_indirect_label.isHidden()
    assert window._sr_axis(spectrum, "f1") is None
    # ...and applying one anyway changes nothing.
    window._apply_reference_shift_hz(100.0, "f1")
    assert "sr_shift_hz_f1" not in spectrum.metadata.raw

    window._on_process_indirect_toggled(True)
    genuine = window.session.spectra[0]
    window._restore_sr_control(genuine)
    assert not window.sr_shift_indirect_spin.isHidden()
    assert not window.sr_indirect_label.isHidden()


def test_main_window_reprocess_reapplies_both_sr_values(qapp, tmp_path, monkeypatch):
    # The post-reprocess reapply path used to restore only the direct
    # dimension's SR, silently dropping the indirect axis's referencing
    # the first time any FID-processing control was touched.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    window._restore_sr_control(spectrum)
    window.sr_shift_spin.setValue(40.0)
    window.sr_shift_indirect_spin.setValue(90.0)
    window._on_apply_sr_clicked()
    f1_referenced = spectrum.axes[0].ppm.copy()
    f2_referenced = spectrum.axes[1].ppm.copy()

    window._apply_loaded_processing_sidecar(spectrum, 0)

    np.testing.assert_allclose(spectrum.axes[0].ppm, f1_referenced)
    np.testing.assert_allclose(spectrum.axes[1].ppm, f2_referenced)


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("dim_index,dim", [(0, "F2"), (1, "F1")])
def test_main_window_phase_settle_preserves_the_2d_zoom(qapp, dim_index, dim, baseline, tmp_path, monkeypatch):
    # Real user report: "Something is up with the GUI phase control for
    # the indirect and it's jumping during phasing." The jump was
    # literal. Two 200 ms debounce timers both restart on every drag
    # tick, so both fire on a pause; _run_debounced_multi_row_reprocess
    # used a bare show_spectrum, which rebuilds the contour and resets
    # the view. A window zoomed to 4..2 x 60..20 ppm snapped back to the
    # full 12.7..-3.3 x 157.5..-7.5 every time the drag paused.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["baseline_correct"] = baseline
    window.dim_combo.setCurrentIndex(dim_index)

    zoom = ((4.0, 2.0), (60.0, 20.0))
    window.canvas.ax.set_xlim(*zoom[0])
    window.canvas.ax.set_ylim(*zoom[1])

    window.phase0_spin.setValue(5.0)
    for timer, run in ((window._live_reprocess_timer, window._run_debounced_multi_row_reprocess),
                       (window._canvas_2d_refresh_timer, window._run_debounced_canvas_2d_refresh)):
        if timer.isActive():
            timer.stop()
            run()

    assert window.canvas.ax.get_xlim() == pytest.approx(zoom[0])
    assert window.canvas.ax.get_ylim() == pytest.approx(zoom[1])


def test_main_window_phase_settle_rebuilds_the_contour_only_once(qapp, tmp_path, monkeypatch):
    # Both debounce timers are 200 ms and both restart on every drag
    # tick, so before this they always fired as a pair on a pause and
    # rebuilt the identical contour twice. Measured at 295 ms per
    # rebuild for Cyclosporine/203 and 1.34 s for the EXSY -- the whole
    # cost of a settle, paid twice for one frame.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8), vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["baseline_correct"] = True
    window.dim_combo.setCurrentIndex(1)

    window.phase0_spin.setValue(5.0)
    assert window._live_reprocess_timer.isActive()

    calls = []
    original = window.canvas.show_spectrum
    window.canvas.show_spectrum = lambda *a, **k: (calls.append(1), original(*a, **k))[1]

    window._live_reprocess_timer.stop()
    window._run_debounced_multi_row_reprocess()
    # The reprocess *is* the settle redraw, so it must also have taken
    # the canvas-refresh timer out of the running.
    assert not window._canvas_2d_refresh_timer.isActive()
    assert window._canvas_2d_refresh_pending_index is None
    assert len(calls) == 1


def test_main_window_first_contour_refresh_debounces_before_any_measurement(qapp, tmp_path):
    # Explicit user choice of adaptive throttling over pure debounce.
    # The first drag on a given shape must still debounce: nothing has
    # been measured yet, and assuming "cheap" would be exactly the
    # wrong guess on the datasets (the EXSY, at 1.34 s a rebuild) that
    # most need protecting from a live redraw per frame.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window._contour_rebuild_seconds.clear()

    window._schedule_canvas_2d_refresh(0)

    assert window._canvas_2d_refresh_timer.isActive()
    assert window._canvas_2d_refresh_timer.interval() == 200


def test_main_window_cheap_contour_refreshes_live_during_a_drag(qapp, tmp_path):
    # A contour already measured as cheap redraws *during* the drag,
    # rather than the user dragging blind until they pause.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    shape = window.session.spectra[0].data.shape
    window._contour_rebuild_seconds[shape] = 0.01  # well under the live limit
    window._last_contour_rebuild_at = 0.0          # long ago -- no throttle wait owed

    calls = []
    original = window._run_debounced_canvas_2d_refresh
    window._run_debounced_canvas_2d_refresh = lambda: (calls.append(1), original())[1]

    window._schedule_canvas_2d_refresh(0)

    # Redrawn immediately, not left on a timer.
    assert calls == [1]
    assert not window._canvas_2d_refresh_timer.isActive()


def test_main_window_expensive_contour_still_debounces(qapp, tmp_path):
    # The EXSY case: a live rebuild would be over a second of frozen UI
    # per frame, strictly worse than showing nothing until the pause.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    shape = window.session.spectra[0].data.shape
    window._contour_rebuild_seconds[shape] = 1.34  # measured EXSY rebuild
    window._last_contour_rebuild_at = 0.0

    calls = []
    original = window._run_debounced_canvas_2d_refresh
    window._run_debounced_canvas_2d_refresh = lambda: (calls.append(1), original())[1]

    window._schedule_canvas_2d_refresh(0)

    assert calls == []
    assert window._canvas_2d_refresh_timer.isActive()
    assert window._canvas_2d_refresh_timer.interval() == 200


def test_main_window_live_contour_refresh_is_throttled_not_per_tick(qapp, tmp_path):
    # "Live" must not mean "rebuild on every drag tick" -- at one
    # rebuild per measured duration the drag's own event handling still
    # gets a share of the main thread.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    shape = window.session.spectra[0].data.shape
    window._contour_rebuild_seconds[shape] = 0.2
    window._last_contour_rebuild_at = time.perf_counter()  # one just happened

    calls = []
    original = window._run_debounced_canvas_2d_refresh
    window._run_debounced_canvas_2d_refresh = lambda: (calls.append(1), original())[1]

    window._schedule_canvas_2d_refresh(0)

    # Deferred rather than run again straight away, and by less than
    # the full 200 ms debounce -- it is waiting out the throttle, not
    # falling back to debounce.
    assert calls == []
    assert window._canvas_2d_refresh_timer.isActive()
    assert 0 < window._canvas_2d_refresh_timer.interval() <= 200


def test_main_window_records_how_long_a_contour_rebuild_took(qapp, tmp_path):
    # The measurement that makes the live-vs-debounce choice
    # self-calibrating rather than a guess from the data's shape.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    spectrum = window.session.spectra[0]
    window._contour_rebuild_seconds.clear()

    window._redraw_2d_preserving_view(spectrum)

    assert spectrum.data.shape in window._contour_rebuild_seconds
    assert window._contour_rebuild_seconds[spectrum.data.shape] >= 0.0


def test_main_window_f2_phase_on_hypercomplex_2d_never_shows_an_invalid_state(qapp, tmp_path, monkeypatch):
    # The hypercomplex decode consumes each row's F2 imaginary part to
    # build F1's, so a hypercomplex 2D's stored complex pair is F1's
    # quadrature. Rotating it by an F2 phase is not an F2 phase
    # correction -- it is the very absorption/dispersion mixing that
    # produced the hourglass lineshape in the first place. So the live
    # tick applies nothing for F2 here and defers to the rebuild, which
    # phases F2 before the decode where it still means something.
    #
    # Asserted as: what the user is shown is always *exactly* a state a
    # clean rebuild can reproduce -- here the last settled phase, to the
    # bit -- rather than an invalid intermediate that happens to be
    # close. Re-enabling the preview (making _f2_quadrature_is_spent
    # return False) breaks this by ~1e-2 relative.
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip("needs the real Cyclosporine/203 dataset")
    _isolate_sidecars(monkeypatch, tmp_path)

    def _build(phases, settle_at):
        window = MainWindow()
        window._load(path, vendor="bruker")
        window._on_process_indirect_toggled(True)
        spectrum = window.session.spectra[0]
        spectrum.metadata.raw["baseline_correct"] = True
        window.dim_combo.setCurrentIndex(0)
        for value in phases:
            window.phase0_spin.setValue(value)
            if value in settle_at and window._live_reprocess_timer.isActive():
                window._live_reprocess_timer.stop()
                window._run_debounced_multi_row_reprocess()
        return np.asarray(spectrum.data)

    # Settle at 30, then keep dragging to 30.5 without settling again.
    shown = _build([30.0, 30.5], {30.0})
    settled = _build([30.0], {30.0})
    np.testing.assert_array_equal(shown, settled)


def test_main_window_f2_quadrature_is_spent_only_for_hypercomplex_2d(qapp, tmp_path, monkeypatch):
    # Modes 0-3 keep F2's own imaginary (their F1 point is one raw
    # complex row, nothing is consumed), and a pseudo-2D never runs an
    # F1 transform at all -- both must keep the fast live F2 preview.
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    spectrum = window.session.spectra[0]
    spectrum.metadata.pseudo_2d = False
    for fn_mode, spent in ((1, False), (3, False), (4, True), (5, True), (6, True)):
        spectrum.metadata.raw["fn_mode"] = fn_mode
        assert window._f2_quadrature_is_spent(spectrum) is spent, fn_mode
    spectrum.metadata.pseudo_2d = True
    spectrum.metadata.raw["fn_mode"] = 6
    assert window._f2_quadrature_is_spent(spectrum) is False


def test_main_window_phase_reference_tracks_what_it_already_contains(qapp, tmp_path, monkeypatch):
    # The bookkeeping the fix above depends on: after a debounced
    # rebuild the reference *is* the rebuilt data, and the phase that
    # data already carries is recorded so the next tick applies only
    # the remainder. Without the second half, the whole phase would be
    # re-applied to data that already had it.
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip("needs the real Cyclosporine/203 dataset")
    _isolate_sidecars(monkeypatch, tmp_path)

    window = MainWindow()
    window._load(path, vendor="bruker")
    window._on_process_indirect_toggled(True)
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["baseline_correct"] = True
    window.dim_combo.setCurrentIndex(1)

    assert window._phase_reference_delta_applied == (0.0, 0.0)
    window.phase0_spin.setValue(30.0)
    window._live_reprocess_timer.stop()
    window._run_debounced_multi_row_reprocess()

    assert window._phase_reference_delta_applied == (30.0, 0.0)
    np.testing.assert_allclose(
        np.real(window._phase_reference_full), np.real(spectrum.complex_data)
    )


def test_main_window_vocs_sum_runs_over_the_list_selection_in_one_step(qapp):
    # "I should be able to select multiple spectra and apply VOCS and
    # sum all in one fell swoop, not apply VOCS per spectrum and then
    # select all and sum." No per-spectrum Apply, no Compare-tab round
    # trip -- select, then one action.
    window = MainWindow()
    for offset in (-2.0, 0.0, 2.0):
        _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=offset))
    window.spectrum_list.selectAll()
    before = window.spectrum_list.count()

    window._on_vocs_sum_selection_action()

    assert window.spectrum_list.count() == before + 1
    derived = window.session.spectra[-1]
    assert derived.metadata.vendor == "derived"
    assert "VOCS sum (3 spectra)" in window.session.label(len(window.session) - 1)


def test_main_window_vocs_sum_selection_needs_at_least_two(qapp, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    window.spectrum_list.selectAll()
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))

    window._on_vocs_sum_selection_action()

    assert shown.get("called") is True
    assert window.spectrum_list.count() == 1   # nothing added


def test_main_window_selected_1d_spectra_skips_2d_and_keeps_list_order(qapp, tmp_path):
    # A VOCS series is 1D by definition; a 2D caught in a drag-select
    # must not be fed to it. Order follows the list, not click order,
    # so the composite is reproducible.
    window = MainWindow()
    for offset in (-2.0, 0.0, 2.0):
        _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse(x0=offset))
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.spectrum_list.selectAll()

    selected = window._selected_1d_spectra()

    assert len(selected) == 3                       # the 2D is skipped
    assert all(s.ndim == 1 for s in selected)
    assert selected == [window.session.spectra[i] for i in range(3)]


# ----------------------------------------------------------------------
# Folding the saved processing into the read (see MainWindow.
# _saved_processing_read_kwargs / _read_already_applied_saved_processing)
# ----------------------------------------------------------------------

#: Saved-processing states and whether reading them should make the
#: row-by-row recompute redundant. The False rows are the two the two
#: pipelines are measured to *disagree* on -- see
#: _read_already_applied_saved_processing's own docstring. They are
#: listed here rather than merely excluded in code so that a later fix
#: to either divergence has a test that notices.
_FOLDED_PROCESSING_CASES = [
    ({}, True),
    ({"phase0": 31.7}, True),
    ({"live_window": "exponential", "live_lb": 15.0, "live_zero_fill": 8192}, True),
    ({"live_window": "gaussian", "live_lb": 5.0, "live_gb": 0.3}, True),
    ({"live_window": "qsine", "live_ssb": 2.0}, True),
    ({"live_window": "sine", "live_ssb": 1.0}, True),
    ({"live_window": "none"}, True),
    ({"live_fid_shift": 12}, True),
    ({"baseline_correct": True, "phase0": 5.0}, True),
    ({"phase0": 10.0, "live_window": "exponential", "live_lb": 20.0}, True),
    ({"phase0": 31.7, "phase1": 44.5}, True),           # exact since _pivot_fraction
    ({"phase1": -25.0, "live_window": "exponential", "live_lb": 20.0}, True),
    ({"live_align_echo": True}, False),                 # reader drops a recorded phase, recompute does not
    ({"phase0_f1": 12.0}, False),                       # no reader parameter for a user F1 phase
]


@pytest.mark.parametrize("saved,expected_fast", _FOLDED_PROCESSING_CASES)
def test_main_window_folded_read_matches_the_recompute_exactly(
    qapp, tmp_path, monkeypatch, saved, expected_fast,
):
    """Reading a 2D already in its saved state must produce *the same
    array* as reading it plain and recomputing every row afterwards.

    This is the correctness half of the load-speed work: skipping the
    recompute is only legitimate while the two agree, so the test
    compares the two directly rather than asserting the fast path was
    taken. Equality is to floating-point exactness (1e-15 of full
    scale), not a loose tolerance -- the same standard the nmrglue
    replacements were held to, and the reason the two known-divergent
    cases above are excluded in code instead of tolerated."""
    source = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "204")
    if not os.path.isdir(source):
        pytest.skip(f"needs the real dataset at {source}")
    path = str(tmp_path / "204")
    shutil.copytree(source, path)
    with open(io_persistence.processing_sidecar_path(path), "w") as f:
        json.dump({"state": saved}, f)

    def load(force_recompute):
        window = MainWindow()
        if force_recompute:
            window._read_already_applied_saved_processing = lambda spectrum: False
        window._load(path, vendor="bruker")
        spectrum = window.session.spectra[0]
        return spectrum.data.copy(), window._read_already_applied_saved_processing(spectrum)

    fast_data, took_fast_path = load(False)
    slow_data, _ = load(True)

    assert took_fast_path is expected_fast
    assert fast_data.shape == slow_data.shape
    scale = np.abs(slow_data).max() or 1.0
    assert np.abs(fast_data - slow_data).max() / scale < 1e-15


def test_main_window_saved_processing_is_folded_into_the_read_kwargs(qapp, tmp_path):
    """The saved state reaches the *reader*, not just the recompute.

    Guards the actual mechanism: if these stopped being passed down,
    every case above would still pass (both paths would simply be slow
    and agree), so the folding itself needs asserting separately."""
    path = _make_1d_dataset(tmp_path)
    with open(io_persistence.processing_sidecar_path(path), "w") as f:
        json.dump({"state": {
            "phase0": 12.5, "phase1": 0.0, "live_window": "exponential",
            "live_lb": 7.0, "live_zero_fill": 4096, "live_fid_shift": 3,
        }}, f)

    kwargs = MainWindow()._initial_load_kwargs(path, "bruker")

    assert kwargs["phase0"] == 12.5
    assert kwargs["zero_fill"] == 4096
    assert kwargs["fid_shift"] == 3
    assert kwargs["window"] == "exponential"
    assert kwargs["window_kwargs"] == {"lb": 7.0}
    # A saved phase is an absolute choice, not a starting point for the
    # reader's own autophase to move away from.
    assert kwargs["autophase"] is False


def test_main_window_unsaved_processing_folds_nothing_into_the_read(qapp, tmp_path):
    """No sidecar means no folded kwargs -- the reader keeps following
    the file's own recorded parameters, exactly as before this
    optimization existed."""
    path = _make_1d_dataset(tmp_path)

    kwargs = MainWindow()._initial_load_kwargs(path, "bruker")

    assert "window" not in kwargs and "phase0" not in kwargs
    assert "autophase" not in kwargs


# ============================================================
# PFG generalisation, guess values, global fit, DOSY
# ============================================================

def _pfg_series_window(qapp, true_D=2.0e-9, n_gradients=14, relative=None,
                        shifts=((5.0, 1.0),)):
    """A MainWindow holding a synthetic pseudo-2D PFG series, one
    Gaussian peak per entry in ``shifts``, all decaying with ``true_D``.
    ``relative`` (0-1 gradient fractions) is stored the way an X-Pulse
    arrayed experiment records its own."""
    from opennmr.processing.fitting import gaussian, lorentzian

    b_values = np.linspace(0.0, 3.0e9, n_gradients)
    ppm = np.linspace(0.0, 10.0, 900)
    fwhm = 0.05
    data = np.zeros((len(b_values), len(ppm)))
    for i, b in enumerate(b_values):
        attenuation = np.exp(-b * true_D)
        for shift, amplitude in shifts:
            data[i] += gaussian(ppm, shift, fwhm, amplitude * attenuation)
    raw = {"gradient_relative": np.asarray(relative, dtype=float)} if relative is not None else {}
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True, raw=raw),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    for shift, _amplitude in shifts:
        window._add_peak_row(shift=shift, fwhm=fwhm, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)
    return window, b_values, spectrum


def test_main_window_b_values_from_gradients_matches_the_diffusion_module(qapp):
    # The PFG generalisation: a real b-value from gradient amplitudes
    # and timings, for a vendor with no diff.xml to read one from.
    relative = np.linspace(0.02, 0.95, 12)
    window, _b, spectrum = _pfg_series_window(qapp, n_gradients=12, relative=relative)
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)
    window.pfg_delta_spin.setValue(2.0)          # ms
    window.pfg_big_delta_spin.setValue(100.0)    # ms
    window.pfg_max_gradient_spin.setValue(55.0)  # G/cm
    window.pfg_shape_combo.setCurrentIndex(window.pfg_shape_combo.findData("smsq10"))
    window.pfg_sequence_combo.setCurrentIndex(window.pfg_sequence_combo.findData("pgse"))

    got = window._resolve_independent_values(spectrum, is_diffusion=True, n_slices=12)
    expected = processing_diffusion.b_values_from_relative(
        relative, 55.0, 2e-3, 100e-3, nucleus="1H", shape="smsq10",
        sequence="pgse", tau=0.0, max_gradient_unit="G/cm")
    np.testing.assert_allclose(got, expected)


def test_main_window_b_from_gradients_needs_a_maximum_gradient(qapp):
    window, _b, spectrum = _pfg_series_window(qapp, n_gradients=6,
                                               relative=np.linspace(0, 1, 6))
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)
    window.pfg_max_gradient_spin.setValue(0.0)
    with pytest.raises(ValueError, match="maximum gradient"):
        window._resolve_independent_values(spectrum, is_diffusion=True, n_slices=6)


def test_main_window_pfg_inputs_only_shown_for_the_gradient_source(qapp):
    window = MainWindow()
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    assert window.pfg_delta_spin.isHidden() is True
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)
    assert window.pfg_delta_spin.isHidden() is False
    assert window.pfg_max_gradient_spin.isHidden() is False


def test_main_window_generalised_diffusion_model_recovers_two_components(qapp):
    from opennmr.processing.fitting import gaussian, lorentzian

    b_values = np.linspace(0.0, 4.0e9, 24)
    ppm = np.linspace(0.0, 10.0, 900)
    data = np.zeros((len(b_values), len(ppm)))
    for i, b in enumerate(b_values):
        amplitude = 0.7 * np.exp(-3e-9 * b) + 0.3 * np.exp(-2e-10 * b)
        data[i] = gaussian(ppm, 5.0, 0.05, amplitude)
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("diffusion_n"))
    window.diffusion_components_spin.setValue(2)

    _fit_spectrum_then_curve(window)

    result = window._relaxation_fit_results[0]
    recovered = sorted([result["params"]["D1"], result["params"]["D2"]], reverse=True)
    assert recovered[0] == pytest.approx(3e-9, rel=0.05)
    assert recovered[1] == pytest.approx(2e-10, rel=0.05)
    # Each component's share of the total, paired with its own D --
    # the form a multi-component result is quoted in.
    report = window.curve_fit_summary_label.text()
    assert "population weights" in report
    assert "%" in report.split("Relative amplitudes")[1]


def test_main_window_component_controls_only_shown_for_the_generalised_model(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    window.relaxation_model_combo.setCurrentIndex(0)  # plain Stejskal-Tanner
    assert window.diffusion_components_spin.isHidden() is True
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("diffusion_n"))
    assert window.diffusion_components_spin.isHidden() is False
    assert window.diffusion_stretched_check.isHidden() is False


# ---- guess values ---------------------------------------------

def test_main_window_guess_table_lists_the_model_parameters(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(0)
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("inversion_recovery"))
    names = [window.guess_table.item(r, 0).text() for r in range(window.guess_table.rowCount())]
    assert names == ["M0", "T1", "C"]

    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("exponential_decay"))
    names = [window.guess_table.item(r, 0).text() for r in range(window.guess_table.rowCount())]
    assert names == ["M0", "T2"]


def test_main_window_guess_table_tracks_the_component_count(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("diffusion_n"))
    window.diffusion_components_spin.setValue(3)
    names = [window.guess_table.item(r, 0).text() for r in range(window.guess_table.rowCount())]
    assert names == ["I1", "D1", "I2", "D2", "I3", "D3"]

    window.diffusion_stretched_check.setChecked(True)
    names = [window.guess_table.item(r, 0).text() for r in range(window.guess_table.rowCount())]
    assert names == ["I1", "D1", "beta1", "I2", "D2", "beta2", "I3", "D3", "beta3"]


def test_main_window_guess_table_is_ignored_while_switched_off(qapp):
    # Off by default -- every existing fit must keep running from the
    # model's own automatic guess exactly as before.
    window = MainWindow()
    assert window.guess_group.isChecked() is False
    assert window._read_guess_table(["M0", "T2"]) == (None, None, None)


def test_main_window_guess_table_supplies_p0_bounds_and_fixed(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(0)
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("inversion_recovery"))
    window.guess_group.setChecked(True)
    # Time rows in this table are in the panel's Unit; these are seconds.
    _set_delay_unit(window, "s")
    window.guess_table.setItem(1, 1, QTableWidgetItem("0.25"))   # T1 guess
    window.guess_table.setItem(1, 2, QTableWidgetItem("0.01"))   # T1 min
    window.guess_table.setItem(1, 3, QTableWidgetItem("5"))      # T1 max
    window.guess_table.item(2, 0).setCheckState(Qt.CheckState.Checked)  # fix C
    window.guess_table.setItem(2, 1, QTableWidgetItem("2.0"))

    p0, bounds, fixed = window._read_guess_table(["M0", "T1", "C"])
    assert p0[1] == 0.25
    assert bounds[0][1] == 0.01 and bounds[1][1] == 5.0
    assert fixed == {"C": 2.0}


def test_main_window_blank_guess_cells_fall_back_to_the_automatic_value(qapp):
    # So one parameter can be pinned without filling in every row.
    window = MainWindow()
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("exponential_decay"))
    window.guess_group.setChecked(True)
    _set_delay_unit(window, "s")
    window.guess_table.setItem(0, 1, QTableWidgetItem(""))
    window.guess_table.setItem(1, 1, QTableWidgetItem("0.07"))
    p0, _bounds, _fixed = window._read_guess_table(["M0", "T2"])
    assert p0[1] == 0.07
    assert np.isfinite(p0[0])


def test_main_window_guess_table_rejects_nonsense_before_the_fit_runs(qapp):
    window = MainWindow()
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("exponential_decay"))
    window.guess_group.setChecked(True)
    _set_delay_unit(window, "s")

    window.guess_table.setItem(1, 1, QTableWidgetItem("not a number"))
    with pytest.raises(ValueError, match="is not a number"):
        window._read_guess_table(["M0", "T2"])

    # A guess outside its own bounds -- scipy rejects this with a far
    # less helpful message, so it is caught here first.
    window.guess_table.setItem(1, 1, QTableWidgetItem("9"))
    window.guess_table.setItem(1, 2, QTableWidgetItem("0.01"))
    window.guess_table.setItem(1, 3, QTableWidgetItem("1"))
    with pytest.raises(ValueError, match="outside its own bounds"):
        window._read_guess_table(["M0", "T2"])


def test_main_window_guess_values_actually_reach_the_fit(qapp):
    window, b_values, _spectrum = _pfg_series_window(qapp, true_D=2.0e-9)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)  # Stejskal-Tanner
    window.guess_group.setChecked(True)
    row = next(r for r in range(window.guess_table.rowCount())
               if window.guess_table.item(r, 0).text() == "D")
    window.guess_table.setItem(row, 1, QTableWidgetItem("1.5e-09"))

    _fit_spectrum_then_curve(window)

    result = window._relaxation_fit_results[0]
    assert result["p0"][row] == pytest.approx(1.5e-9)
    assert result["params"]["D"] == pytest.approx(2.0e-9, rel=0.05)


def test_main_window_adopt_last_fit_fills_the_guess_column(qapp):
    window, b_values, _spectrum = _pfg_series_window(qapp)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)
    _fit_spectrum_then_curve(window)

    window._on_adopt_fit_as_guess_clicked()
    assert window.guess_group.isChecked() is True
    fitted = window._relaxation_fit_results[0]["params"]
    for row in range(window.guess_table.rowCount()):
        name = window.guess_table.item(row, 0).text()
        assert float(window.guess_table.item(row, 1).text()) == pytest.approx(
            fitted[name], rel=1e-4)


# ---- reporting -------------------------------------------------

def test_main_window_curve_fit_report_states_the_uncertainty_kind(qapp):
    window, b_values, _spectrum = _pfg_series_window(qapp)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)
    _fit_spectrum_then_curve(window)

    # A single peak gets the full report -- units, uncertainty kind and
    # quality, not a bare "D = 2e-09 ± 5e-11".
    report = window.curve_fit_summary_label.text()
    assert "Uncertainty" in report and "1σ" in report
    assert "R²" in report
    # And the Fit tab's own panel, sitting right under the curve.
    assert "Uncertainty" in window.relaxation_fit_params_label.text()


def test_main_window_multi_peak_summary_names_the_uncertainty_once(qapp):
    window, b_values, _spectrum = _pfg_series_window(
        qapp, shifts=((3.0, 1.0), (7.0, 0.6)))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)
    _fit_spectrum_then_curve(window)

    summary = window.curve_fit_summary_label.text()
    assert "2 peaks" in summary
    assert summary.count("± is") == 1
    assert "m^2/s" in summary and "R²=" in summary


def test_main_window_fit_report_csv_carries_the_uncertainty_kind_and_quality(qapp):
    window, b_values, _spectrum = _pfg_series_window(qapp)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)
    _fit_spectrum_then_curve(window)

    report = window._build_fit_report_csv(window._slice_fit_results,
                                           window._relaxation_fit_results)
    assert "Uncertainty:" in report
    assert "R^2" in report and "reduced chi^2" in report


# ---- global (multivariate) fit ---------------------------------

def test_main_window_global_fit_ties_one_d_across_every_peak(qapp):
    window, b_values, _spectrum = _pfg_series_window(
        qapp, true_D=1.2e-9, shifts=((3.0, 1.0), (5.0, 0.6), (7.0, 0.3)))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)
    window.global_fit_check.setChecked(True)
    assert window.shared_params_edit.text() == "D"  # prefilled with the rate constant

    _fit_spectrum_then_curve(window)

    assert window._global_fit_result is not None
    assert window._global_fit_result["shared"]["D"] == pytest.approx(1.2e-9, rel=0.05)
    values = {r["params"]["D"] for r in window._relaxation_fit_results}
    assert len(values) == 1
    assert "Shared across every curve" in window.curve_fit_summary_label.text()


def test_main_window_global_fit_needs_a_shared_parameter_named(qapp):
    window, b_values, _spectrum = _pfg_series_window(qapp)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.global_fit_check.setChecked(True)
    window.shared_params_edit.setText("")
    window._on_fit_spectrum_clicked()
    with pytest.raises(ValueError, match="at least one shared parameter"):
        window._run_global_curve_fit(window._slice_fit_results, b_values,
                                      "stejskal_tanner", None, None, None, None)


def test_main_window_shared_field_only_shown_when_global_fitting(qapp):
    window = MainWindow()
    assert window.shared_params_edit.isHidden() is True
    window.global_fit_check.setChecked(True)
    assert window.shared_params_edit.isHidden() is False


# ---- DOSY ------------------------------------------------------

def test_main_window_dosy_map_places_peaks_at_their_own_d(qapp, monkeypatch):
    from opennmr.processing.fitting import gaussian, lorentzian

    b_values = np.linspace(0.0, 4.0e9, 24)
    ppm = np.linspace(0.0, 10.0, 400)
    data = np.zeros((len(b_values), len(ppm)))
    for i, b in enumerate(b_values):
        data[i] += gaussian(ppm, 3.0, 0.08, 1.0 * np.exp(-1.5e-9 * b))
        data[i] += gaussian(ppm, 7.0, 0.08, 1.0 * np.exp(-1.2e-10 * b))
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    monkeypatch.setattr(_DosySettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    window._on_dosy_clicked()

    dosy = window._dosy_result
    assert dosy is not None
    for shift, true_d in ((3.0, 1.5e-9), (7.0, 1.2e-10)):
        column = int(np.argmin(np.abs(ppm - shift)))
        found = dosy["D"][int(np.argmax(dosy["Z"][:, column]))]
        assert found == pytest.approx(true_d, rel=0.3)
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    # The regularisation weight is stated: an ILT's peak width is set by
    # it, so a summary quoting a spread without it reports a setting as
    # a measurement.
    assert "regularisation α" in window.dosy_results_label.text()


def test_main_window_dosy_refuses_a_slice_index_axis(qapp, monkeypatch):
    # Falling back to 1, 2, 3, ... would put a "D" axis in no physical
    # unit on a plot labelled m²/s.
    window, _b, _spectrum = _pfg_series_window(qapp, n_gradients=8)
    window.delay_source_combo.setCurrentIndex(0)  # Auto -> no diff.xml -> slice index
    warned = {}
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warned.setdefault("title", args[1]))
    monkeypatch.setattr(_DosySettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_dosy_clicked()
    assert warned.get("title") == "No real b-values"
    assert window._dosy_result is None


def test_main_window_dosy_needs_a_pseudo_2d(qapp, monkeypatch, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    seen = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: seen.setdefault("title", args[1]))
    window._on_dosy_clicked()
    assert seen.get("title") == "Not supported"


def test_dosy_settings_dialog_reports_its_own_values(qapp):
    dialog = _DosySettingsDialog()
    dialog.auto_alpha_check.setChecked(False)
    dialog.alpha_spin.setValue(0.005)
    dialog.d_min_spin.setValue(-12.0)
    dialog.d_max_spin.setValue(-9.0)
    dialog.threshold_spin.setValue(5.0)
    values = dialog.values()
    assert values["alpha"] == pytest.approx(0.005)
    assert values["d_min"] == pytest.approx(1e-12)
    assert values["d_max"] == pytest.approx(1e-9)
    assert values["threshold"] == pytest.approx(0.05)
    assert values["auto_alpha"] is False


# ---- gradient calibration --------------------------------------

def test_main_window_gradient_calibration_recovers_a_known_maximum(qapp, monkeypatch):
    known_d = 2.299e-9
    relative = np.linspace(0.02, 0.95, 16)
    true_g_max_g_per_cm = 55.0
    b = processing_diffusion.b_values_from_relative(
        relative, true_g_max_g_per_cm, 2e-3, 100e-3, nucleus="1H", shape="smsq10",
        max_gradient_unit="G/cm")

    from opennmr.processing.fitting import gaussian, lorentzian
    ppm = np.linspace(0.0, 10.0, 700)
    data = np.array([gaussian(ppm, 5.0, 0.05, np.exp(-known_d * bi)) for bi in b])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True,
                              raw={"gradient_relative": relative}),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._on_fit_spectrum_clicked()

    monkeypatch.setattr(_GradientCalibrationDialog, "exec",
                        lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    window._on_calibrate_gradient_clicked()

    assert window.pfg_max_gradient_spin.value() == pytest.approx(true_g_max_g_per_cm, rel=0.02)
    assert "Maximum gradient" in window.gradient_calibration_label.text()


def test_main_window_gradient_calibration_needs_a_spectral_fit_first(qapp, monkeypatch):
    window, _b, _spectrum = _pfg_series_window(qapp, n_gradients=8)
    seen = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: seen.setdefault("title", args[1]))
    window._on_calibrate_gradient_clicked()
    assert seen.get("title") == "Nothing to calibrate against"


def test_gradient_calibration_dialog_prefills_from_the_analysis_tab(qapp):
    dialog = _GradientCalibrationDialog(pfg_settings={
        "delta": 0.003, "DELTA": 0.15, "tau": 0.002,
        "shape": "sine", "sequence": "bipolar", "max_gradient_g_per_cm": 42.0,
    })
    values = dialog.values()
    assert values["delta"] == pytest.approx(0.003)
    assert values["DELTA"] == pytest.approx(0.15)
    assert values["tau"] == pytest.approx(0.002)
    assert values["shape"] == "sine"
    assert values["sequence"] == "bipolar"
    # The standards list leads with a real literature value, not a
    # blank to be typed in.
    assert values["known_D"] == pytest.approx(2.299e-9)


# ============================================================
# Per-peak curve-fit guesses, staged-measurement provenance,
# and the Spectral Fitting tab split
# ============================================================

def test_main_window_fitting_tab_has_three_collapsible_sections(qapp):
    # "Fitting spectra should be its own tab" -- Peak Picking and
    # Integration were sub-tabs of Analysis sitting beside Curve
    # Fitting, which put a prerequisite and its consumer at the same
    # level and hid the order between them. Later folded into three
    # independent collapsible sections of one "Fitting" launcher
    # button (replacing the standalone "Spectral Fitting" tab this
    # test used to check for) -- explicit user request, also pulling
    # in CSA (renamed "CSA/Dipolar"), previously a fourth Analysis
    # sub-tab.
    window = MainWindow()
    side = _side_tab_names(window)
    assert "Fitting" in side
    assert "Spectral Fitting" not in side
    assert "Analysis" not in side
    # Step 1 (measure) before step 2 (Relaxation's curve fit).
    assert side.index("Fitting") < side.index("Relaxation")

    fitting_tab = window.side_tabs.widget(window._fitting_tab_index)
    sections = fitting_tab.findChildren(CollapsibleSection)
    titles = [s.title() for s in sections]
    # J-coupling between Integration and CSA/Dipolar (test_gui_jcoupling.py);
    # Quadrupolar (test_gui_quadrupolar_fit.py), then Combined
    # Hamiltonian (test_gui_combined_fit.py), Multi-spectrum last
    # (test_gui_multi_fit.py). Quadrupolar (Czjzek distribution) right
    # after Quadrupolar, and Fit plot display options at the very end
    # (test_gui_czjzek_display.py).
    assert titles == ["Isotropic (Gauss/Lor)", "Integration", "J-coupling", "CSA/Dipolar",
                      "Quadrupolar", "Quadrupolar (Czjzek distribution)", "Combined Hamiltonian",
                      "Multi-spectrum", "Kept fit models", "Fit plot: components and colours"]

    # eNMR/Imaging moved off this tab entirely, to their own
    # "Applications" button (see
    # test_main_window_applications_is_its_own_top_level_button).
    applications_tab = window.side_tabs.widget(side.index("Applications"))
    analysis = [s.title() for s in applications_tab.findChildren(CollapsibleSection)]
    # ... and DNP joined them (2026-09-30), with its two sub-sections.
    assert analysis == ["eNMR", "Imaging", "DNP", "Enhancement", "Field Sweep Profile"]


def test_main_window_advanced_processing_is_its_own_top_level_button(qapp):
    # Explicit user request: seven launcher buttons replacing the old
    # Data/Processing/Spectral Fitting/Analysis/Compare tabs, one of
    # which is "Advanced Processing", which "for now, is just the
    # 'VOCS' processing, which is currently in 'Analysis'". VOCS is
    # therefore no longer a sub-tab of Analysis (see
    # test_main_window_fitting_tab_has_three_collapsible_sections,
    # just above, for the rest of that sub-tab list).
    window = MainWindow()
    side = _side_tab_names(window)
    assert "Advanced Processing" in side
    # Comes right after Processing, ahead of Fitting/Analysis --
    # Toolbar order (explicit user request, 2026-09-27): Open, Process,
    # Compare, Fit, Simulate, Dynamics, Adv. Proc., Apply, Settings.
    assert side.index("Fitting") < side.index("Relaxation") < side.index("Advanced Processing")
    assert side.index("Advanced Processing") < side.index("Applications")

    # VOCS belongs only here -- not on any section of the Applications
    # tab (the old combined "Analysis" tab no longer exists at all).
    applications_tab = window.side_tabs.widget(side.index("Applications"))
    application_sections = [s.title() for s in applications_tab.findChildren(CollapsibleSection)]
    assert "VOCS" not in application_sections

    # The VOCS controls themselves are unmoved code (same _build_vocs_
    # subtab widget, same attribute names) -- just re-parented onto
    # the new top-level tab instead of a sub-tab strip.
    advanced_processing_tab = window.side_tabs.widget(side.index("Advanced Processing"))
    assert advanced_processing_tab.isAncestorOf(window.vocs_pulse_edit)
    assert advanced_processing_tab.isAncestorOf(window.vocs_window_filter_checkbox)


def test_main_window_relaxation_is_its_own_top_level_button(qapp):
    # Explicit user request: "this is where the 'Curve fitting' (in
    # the current 'Analysis' tab) should go" -- Curve Fitting is no
    # longer a sub-tab of Analysis, and now has its own "Relaxation"
    # launcher button (the sole content, like Advanced Processing's
    # own VOCS-only panel).
    window = MainWindow()
    side = _side_tab_names(window)
    assert "Relaxation" in side
    # Comes right after Fitting, ahead of the last button,
    # Applications -- the order the seven launcher buttons were
    # given in.
    assert side.index("Fitting") < side.index("Relaxation")
    assert side.index("Relaxation") < side.index("Applications")

    # Curve Fitting belongs only here -- not on any section of the
    # Applications tab (the old combined "Analysis" tab no longer
    # exists at all).
    applications_tab = window.side_tabs.widget(side.index("Applications"))
    application_sections = [s.title() for s in applications_tab.findChildren(CollapsibleSection)]
    assert "Curve Fitting" not in application_sections

    # Same unmoved widgets/attributes as Curve Fitting always had --
    # just re-parented onto the new top-level tab.
    relaxation_tab = window.side_tabs.widget(side.index("Relaxation"))
    assert relaxation_tab.isAncestorOf(window.fit_type_combo)
    assert relaxation_tab.isAncestorOf(window.relaxation_model_combo)
    assert relaxation_tab.isAncestorOf(window.analysis_status_label)
    assert not applications_tab.isAncestorOf(window.fit_type_combo)


def test_main_window_applications_is_its_own_top_level_button(qapp):
    # Explicit user request: "this is where the eNMR and Imaging
    # sections (currently in the 'Analysis' tab) should go; eNMR and
    # Imaging should be their own separate collapsable sections" --
    # the last of the seven launcher buttons, and the point at which
    # the old combined "Analysis" tab (and its analysis_sub_tabs
    # strip) is fully retired: nothing is built from it any more.
    window = MainWindow()
    side = _side_tab_names(window)
    assert "Applications" in side
    assert "Analysis" not in side
    assert not hasattr(window, "analysis_sub_tabs")
    # Last of the seven, right after Relaxation.
    assert side.index("Relaxation") < side.index("Applications")

    applications_tab = window.side_tabs.widget(side.index("Applications"))
    sections = applications_tab.findChildren(CollapsibleSection)
    assert [s.title() for s in sections] == ["eNMR", "Imaging", "DNP", "Enhancement",
                                             "Field Sweep Profile"]

    enmr_section = _enclosing_collapsible_section(window.enmr_voltage_label)
    imaging_section = _enclosing_collapsible_section(window.imaging_max_gradient_spin)
    assert enmr_section.title() == "eNMR"
    assert imaging_section.title() == "Imaging"
    assert enmr_section is not imaging_section
    assert not enmr_section.isAncestorOf(window.imaging_max_gradient_spin)
    assert not imaging_section.isAncestorOf(window.enmr_voltage_label)

    # Same unmoved widgets/attributes as eNMR/Imaging always had --
    # just re-parented onto the new top-level tab.
    assert applications_tab.isAncestorOf(window.enmr_voltage_label)
    assert applications_tab.isAncestorOf(window.imaging_max_gradient_spin)


# ============================================================
# Fitting tab: "Export ... to Pseudo-2D..." (fitted/integrated
# series reconstructed and stacked into a new pseudo-2D dataset)
# ============================================================

def test_main_window_export_buttons_start_disabled(qapp):
    # A single 1D spectrum, or nothing selected at all, is not a
    # series -- both export buttons should stay off until a
    # multi-slice Fit Spectrum/Integrate Regions has actually run.
    window = MainWindow()
    assert window.export_isotropic_pseudo2d_button.isEnabled() is False
    assert window.export_integration_pseudo2d_button.isEnabled() is False
    assert window.export_csa_pseudo2d_button.isEnabled() is False

    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    window._on_fit_spectrum_clicked()  # a single 1D spectrum -- one slice, not a series
    assert window.export_isotropic_pseudo2d_button.isEnabled() is False


def test_main_window_export_isotropic_pseudo2d_reconstructs_fitted_series(qapp):
    # Explicit user request: fitted peaks exported to a pseudo-2D
    # dataset, so a fitted series can be viewed/compared exactly like
    # the raw data it came from. Reconstructs each slice's summed
    # fitted lineshape (Fit Spectrum's own y_fit).
    delays = np.geomspace(0.001, 0.3, 5)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._on_fit_spectrum_clicked()
    assert window.export_isotropic_pseudo2d_button.isEnabled() is True

    n_before = len(window.session.spectra)
    window._on_export_isotropic_pseudo2d_clicked()
    assert len(window.session.spectra) == n_before + 1

    exported = window.session.spectra[-1]
    assert exported.is_pseudo_2d is True
    assert exported.data.shape == (5, spectrum.data.shape[1])
    np.testing.assert_allclose(exported.axes[1].ppm, spectrum.axes[1].ppm)
    # A genuine reconstruction, not a copy of the raw data -- the
    # fitted lineshape is smooth/noise-free where the raw data isn't.
    assert not np.allclose(exported.data, spectrum.data)


def test_main_window_export_integration_pseudo2d_masks_outside_regions(qapp):
    # Integration's own counterpart -- no fitted lineshape exists, so
    # this stacks each slice's own data with everything outside the
    # integrated region(s) masked to zero, showing exactly what
    # contributed to that slice's integral.
    delays = np.geomspace(0.001, 0.3, 5)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05, n_pts=200)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_region_row(4.5, 5.5)
    window._on_integrate_regions_clicked()
    assert window.export_integration_pseudo2d_button.isEnabled() is True
    # Fitting a different kind of result afterwards must not leave
    # the wrong button enabled -- they read the same staged slot.
    assert window.export_isotropic_pseudo2d_button.isEnabled() is False

    n_before = len(window.session.spectra)
    window._on_export_integration_pseudo2d_clicked()
    assert len(window.session.spectra) == n_before + 1

    exported = window.session.spectra[-1]
    assert exported.is_pseudo_2d is True
    ppm = exported.axes[1].ppm
    outside = (ppm < 4.5) | (ppm > 5.5)
    assert np.allclose(exported.data[:, outside], 0.0)
    assert not np.allclose(exported.data[:, ~outside], 0.0)


def test_main_window_export_csa_pseudo2d_stays_disabled(qapp, monkeypatch):
    # The CSA export turns on only for a CSA series (Fit Every Slice with
    # the CSA model -- tests/test_series_fitting.py), not because some
    # other kind of fit (here an isotropic peak fit) was run.
    delays = np.geomspace(0.001, 0.3, 5)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._on_fit_spectrum_clicked()
    assert window.export_csa_pseudo2d_button.isEnabled() is False
    # A documented no-op, not a silent one, if ever triggered anyway.
    shown = {}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.setdefault("called", True))
    window._on_export_csa_pseudo2d_clicked()
    assert shown.get("called") is True


def test_main_window_curve_fit_names_what_it_will_fit(qapp, tmp_path):
    # "It isn't clear if the fitted spectra are being used in the curve
    # fits." They are; nothing said so.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=64), vendor="bruker")
    assert "No measured intensities yet" in window.curve_fit_source_label.text()
    assert "never re-fits peaks itself" in window.curve_fit_source_label.text()

    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.0)
    window._on_fit_spectrum_clicked()
    staged = window.curve_fit_source_label.text()
    assert "Will fit:" in staged
    assert "1 peak" in staged and "4 slices" in staged
    assert "peak fit" in staged


def test_main_window_staged_measurement_names_region_integration(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=3, n_f2=64), vendor="bruker")
    window._add_region_row(4.0, 6.0)
    window._on_integrate_regions_clicked()
    staged = window.curve_fit_source_label.text()
    assert "region integration" in staged
    assert "1 region" in staged and "3 slices" in staged


def test_main_window_guess_peak_combo_lists_every_peak(qapp):
    window = MainWindow()
    window._add_peak_row(shift=3.2, fwhm=0.1, eta=0.0)
    window._add_peak_row(shift=7.4, fwhm=0.1, eta=0.0)
    window.guess_group.setChecked(True)
    labels = [window.guess_peak_combo.itemText(i) for i in range(window.guess_peak_combo.count())]
    assert labels[0] == "All peaks"
    # Named by shift -- "Peak 2" alone is not something anyone can match
    # against a spectrum.
    assert "3.2" in labels[1] and "7.4" in labels[2]
    assert [window.guess_peak_combo.itemData(i) for i in range(3)] == [None, 0, 1]


def test_main_window_per_peak_guess_overrides_only_the_chosen_peak(qapp):
    window = MainWindow()
    window._add_peak_row(shift=3.2, fwhm=0.1, eta=0.0)
    window._add_peak_row(shift=7.4, fwhm=0.1, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)
    window.relaxation_model_combo.setCurrentIndex(0)  # Stejskal-Tanner
    window.guess_group.setChecked(True)

    d_row = next(r for r in range(window.guess_table.rowCount())
                 if window.guess_table.item(r, 0).text() == "D")
    # Shared set first.
    window.guess_table.setItem(d_row, 1, QTableWidgetItem("1e-09"))
    # Then peak 2 alone.
    window.guess_peak_combo.setCurrentIndex(2)
    window.guess_table.setItem(d_row, 1, QTableWidgetItem("5e-11"))

    p0, _bounds, _fixed = window._read_peak_guess_tables(["I0", "D"], n_peaks=2)
    assert isinstance(p0, dict)
    assert p0[0][d_row] == pytest.approx(1e-9)   # peak 1 keeps the shared guess
    assert p0[1][d_row] == pytest.approx(5e-11)  # peak 2 has its own


def test_main_window_clicking_through_peaks_does_not_pin_them(qapp):
    # Merely selecting a peak must not freeze it against a later change
    # to the shared set -- otherwise browsing the combo silently makes
    # every peak an override.
    window = MainWindow()
    window._add_peak_row(shift=3.2, fwhm=0.1, eta=0.0)
    window._add_peak_row(shift=7.4, fwhm=0.1, eta=0.0)
    window.guess_group.setChecked(True)
    window.guess_peak_combo.setCurrentIndex(1)
    window.guess_peak_combo.setCurrentIndex(2)
    window.guess_peak_combo.setCurrentIndex(0)
    assert [key for key in window._guess_overrides if key is not None] == []


def test_main_window_clear_peak_override_returns_it_to_the_shared_set(qapp):
    window = MainWindow()
    window._add_peak_row(shift=3.2, fwhm=0.1, eta=0.0)
    window._add_peak_row(shift=7.4, fwhm=0.1, eta=0.0)
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("exponential_decay"))
    window.guess_group.setChecked(True)
    _set_delay_unit(window, "s")
    window.guess_table.setItem(1, 1, QTableWidgetItem("0.02"))
    window.guess_peak_combo.setCurrentIndex(2)
    window.guess_table.setItem(1, 1, QTableWidgetItem("0.9"))
    window.guess_peak_combo.setCurrentIndex(0)
    assert 1 in window._guess_overrides

    window.guess_peak_combo.setCurrentIndex(2)
    window.clear_peak_guess_button.click()
    assert 1 not in window._guess_overrides
    assert window.guess_table.item(1, 1).text() == "0.02"  # back on the shared set


def test_main_window_per_peak_guesses_reach_the_fit(qapp):
    # Two peaks with genuinely different diffusion coefficients -- a
    # single shared starting D is a poor guess for at least one of them,
    # which is the case per-peak overrides exist for.
    from opennmr.processing.fitting import gaussian, lorentzian

    b_values = np.linspace(0.0, 4.0e9, 20)
    ppm = np.linspace(0.0, 10.0, 900)
    data = np.zeros((len(b_values), len(ppm)))
    for i, b in enumerate(b_values):
        data[i] += gaussian(ppm, 3.0, 0.05, np.exp(-3.0e-9 * b))
        data[i] += gaussian(ppm, 7.0, 0.05, np.exp(-8.0e-11 * b))
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=3.0, fwhm=0.05, eta=0.0)
    window._add_peak_row(shift=7.0, fwhm=0.05, eta=0.0)
    window.fit_type_combo.setCurrentIndex(1)
    window.relaxation_model_combo.setCurrentIndex(0)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.guess_group.setChecked(True)
    d_row = next(r for r in range(window.guess_table.rowCount())
                 if window.guess_table.item(r, 0).text() == "D")
    window.guess_table.setItem(d_row, 1, QTableWidgetItem("3e-09"))
    window.guess_peak_combo.setCurrentIndex(2)
    window.guess_table.setItem(d_row, 1, QTableWidgetItem("8e-11"))

    _fit_spectrum_then_curve(window)

    results = window._relaxation_fit_results
    assert results[0]["params"]["D"] == pytest.approx(3.0e-9, rel=0.1)
    assert results[1]["params"]["D"] == pytest.approx(8.0e-11, rel=0.1)
    # Each peak actually started from its own guess, not a shared one.
    assert results[0]["p0"][d_row] == pytest.approx(3e-9)
    assert results[1]["p0"][d_row] == pytest.approx(8e-11)


def test_main_window_anchor_slice_reaches_the_spectral_fit(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=5, n_f2=64), vendor="bruker")
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.0)
    window.anchor_slice_spin.setValue(3)

    captured = {}
    real = processing_fitting.NMR2D_fit_slices

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    processing_fitting.NMR2D_fit_slices = spy
    try:
        window._on_fit_spectrum_clicked()
    finally:
        processing_fitting.NMR2D_fit_slices = real
    assert captured["seed_mode"] == "apply_first"
    assert captured["anchor_slice"] == 3


def test_main_window_no_anchor_passed_for_a_single_1d_spectrum(qapp):
    # A 1D spectrum has no other slice to take a lineshape from, so
    # naming an anchor would describe a choice that isn't there.
    window = MainWindow()
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)

    captured = {}
    real = processing_fitting.NMR2D_fit_slices

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    processing_fitting.NMR2D_fit_slices = spy
    try:
        window._on_fit_spectrum_clicked()
    finally:
        processing_fitting.NMR2D_fit_slices = real
    assert captured["anchor_slice"] is None


# ============================================================
# SCORE (multivariate component resolution)
# ============================================================

def _overlapping_pfg_window(qapp, n_levels=20):
    """A MainWindow holding a synthetic two-species PFG series whose
    peaks overlap at 5.0 ppm -- the case a per-peak fit cannot resolve
    and SCORE can."""
    from opennmr.processing.fitting import lorentzian

    b = np.linspace(0.0, 3.0e9, n_levels)
    ppm = np.linspace(0.0, 10.0, 1200)
    fast = (lorentzian(ppm, 2.0, 0.05, 1.0) + lorentzian(ppm, 5.0, 0.05, 0.8)
            + lorentzian(ppm, 8.2, 0.05, 0.5))
    slow = lorentzian(ppm, 5.0, 0.05, 0.9) + lorentzian(ppm, 6.5, 0.05, 1.1)
    data = (np.outer(np.exp(-b * 1.8e-9), fast) + np.outer(np.exp(-b * 2.5e-10), slow))
    data = data + np.random.default_rng(0).normal(0, 2e-3, data.shape)

    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, n_levels + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    window.manual_delays_edit.setText(", ".join(f"{v:.8g}" for v in b))
    return window, b, ppm


def test_main_window_score_resolves_overlapping_species(qapp, monkeypatch):
    window, _b, ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    window._on_score_clicked()

    result = window._score_result
    assert result is not None
    assert result["D"][0] == pytest.approx(2.5e-10, rel=0.1)
    assert result["D"][1] == pytest.approx(1.8e-9, rel=0.1)

    # The peak table was never touched -- SCORE does not use it, which
    # is the point, and the results text says so.
    assert window.peak_table.rowCount() == 0
    assert "peak table" in window.score_results_label.text()
    assert "SCORE" in window.score_results_label.text()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_main_window_score_uses_the_curve_fit_panels_model(qapp, monkeypatch):
    # "the SCORE methods should utilise the same fitting functions as
    # the model for the curve fit" -- SCORE now evaluates whatever
    # RELAXATION_MODELS entry the Model combo names, not a private
    # exponential of its own.
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    index = window.relaxation_model_combo.findData("stejskal_tanner_stretched")
    assert index >= 0
    window.relaxation_model_combo.setCurrentIndex(index)

    window._on_score_clicked()

    result = window._score_result
    assert result["model"] == "stejskal_tanner_stretched"
    # Each component carries the model's own extra parameter.
    assert all("beta" in entry for entry in result["component_params"])
    assert "stejskal_tanner_stretched" in window.score_results_label.text()


def test_main_window_score_default_model_is_the_plain_stejskal_tanner(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    assert window._score_result["model"] == "stejskal_tanner"


def test_main_window_score_rejects_a_multi_component_model_with_the_reason(qapp, monkeypatch):
    # A biexponential already describes the populations SCORE's own
    # component count describes; asking for both is asking for four.
    # Reported against the Model combo, before the settings dialog.
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    index = window.relaxation_model_combo.findData("stejskal_tanner_biexponential")
    assert index >= 0
    window.relaxation_model_combo.setCurrentIndex(index)

    shown = {}
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *args, **kwargs: shown.update(title=args[1], text=args[2]))
    monkeypatch.setattr(_ScoreSettingsDialog, "exec",
                        lambda self: pytest.fail("the settings dialog should not open"))

    window._on_score_clicked()

    assert "no single amplitude parameter" in shown["text"]
    assert window._score_result is None


def test_main_window_score_dialog_names_the_model_it_will_use(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    seen = {}

    def capture(self):
        seen["labels"] = [w.text() for w in self.findChildren(QLabel)]
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(_ScoreSettingsDialog, "exec", capture)
    window._on_score_clicked()
    assert any("Curve Fitting sub-tab" in text for text in seen["labels"])


def test_main_window_score_dialog_prefills_the_svd_suggestion(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    seen = {}

    def capture(self):
        seen["suggested"] = self.components_spin.value()
        seen["text"] = self.suggestion_label.text()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(_ScoreSettingsDialog, "exec", capture)
    window._on_score_clicked()
    assert seen["suggested"] == 2
    # Presented as a suggestion, since the component count is an input
    # the method cannot supply.
    assert "suggest" in seen["text"].lower()
    assert window._score_result is None  # cancelled, nothing run


def test_main_window_score_needs_a_pseudo_2d(qapp, monkeypatch, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    seen = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: seen.setdefault("title", args[1]))
    window._on_score_clicked()
    assert seen.get("title") == "Not supported"


def test_main_window_score_refuses_a_slice_index_axis(qapp, monkeypatch):
    # Without real b-values the "diffusion coefficients" would be in no
    # physical unit at all.
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    window.delay_source_combo.setCurrentIndex(0)  # Auto -> slice index
    warned = {}
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warned.setdefault("title", args[1]))
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    assert warned.get("title") == "No real b-values"
    assert window._score_result is None


def test_score_settings_dialog_reports_its_own_values(qapp):
    dialog = _ScoreSettingsDialog(suggested=3)
    assert dialog.components_spin.value() == 3
    dialog.threshold_spin.setValue(5.0)
    dialog.non_negative_check.setChecked(False)
    values = dialog.values()
    assert values["n_components"] == 3
    assert values["threshold"] == pytest.approx(0.05)
    assert values["non_negative"] is False
    # The strict option is meaningless with the constraint off.
    assert values["non_negative_during_fit"] is False


def test_main_window_score_uses_the_compare_style_stacked_view(qapp, monkeypatch):
    # Explicit user request: the output "should look like the compare
    # tab where we have multiple selectable spectra". Rendered through
    # the same MplCanvas.show_compare, so each component gets its own
    # axes with its own y-zoom, exactly as a Compare stack does.
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()

    assert window.fit_canvas._compare_axes is not None
    assert len(window.fit_canvas._compare_axes) == 2   # both components, measured unchecked
    for axes in window.fit_canvas._compare_axes:
        assert len(axes.lines) == 1


def test_main_window_score_component_list_is_selectable(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()

    listing = window.score_component_list
    assert listing.isHidden() is False
    assert listing.count() == 3            # two components plus "Measured"
    # Components on by default; the measured spectrum off, since it is
    # for comparison rather than part of the answer.
    assert listing.item(0).checkState() == Qt.CheckState.Checked
    assert listing.item(2).checkState() == Qt.CheckState.Unchecked
    # Each row names its own diffusion coefficient, not just a number.
    assert "m^2/s" in listing.item(0).text()
    assert "Measured" in listing.item(2).text()


def test_main_window_unchecking_a_component_drops_it_from_the_plot(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    assert len(window.fit_canvas._compare_axes) == 2

    window.score_component_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    # A single remaining trace is drawn on one axes, not a stack of
    # one -- show_compare's own long-standing behaviour.
    assert window.fit_canvas._compare_axes is None
    assert len(window.fit_canvas.ax.lines) == 1

    # And the measured spectrum can be brought in for comparison.
    window.score_component_list.item(2).setCheckState(Qt.CheckState.Checked)
    assert len(window.fit_canvas._compare_axes) == 2


def test_main_window_selecting_a_component_row_highlights_it(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()

    window.score_component_list.setCurrentRow(1)
    assert window.fit_canvas._selected_compare_index == 1

    # A row that is not plotted cannot be highlighted -- the highlight
    # is indexed among the *checked* rows, not among every listed one.
    window.score_component_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.score_component_list.setCurrentRow(0)
    assert window.fit_canvas._selected_compare_index is None


def _distinct_similar_pfg_window(qapp, ratio=1.05, n_levels=24):
    """A series of two species with **distinct** peaks and nearly equal
    diffusion coefficients -- the case OUTSCORE exists for."""
    from opennmr.processing.fitting import lorentzian

    ppm = np.linspace(0.0, 10.0, 1500)
    a = (lorentzian(ppm, 1.2, 0.04, 1.0) + lorentzian(ppm, 3.4, 0.04, 0.7)
         + lorentzian(ppm, 7.1, 0.04, 0.5))
    c = (lorentzian(ppm, 2.3, 0.04, 0.8) + lorentzian(ppm, 5.6, 0.04, 1.0)
         + lorentzian(ppm, 8.4, 0.04, 0.6))
    slow = 1.0e-9
    b = np.linspace(0.0, 3.0e9, n_levels)
    data = np.outer(np.exp(-b * slow * ratio), a) + np.outer(np.exp(-b * slow), c)
    data = data + np.random.default_rng(0).normal(0, 1.5e-3, data.shape)

    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, n_levels + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{v:.8g}" for v in b))
    return window, slow, slow * ratio


def _accept_score_dialog(monkeypatch, method="score"):
    def accept(self):
        self.method_combo.setCurrentIndex(self.method_combo.findData(method))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(_ScoreSettingsDialog, "exec", accept)


def test_main_window_outscore_separates_similar_sized_species(qapp, monkeypatch):
    window, slow, fast = _distinct_similar_pfg_window(qapp)
    _accept_score_dialog(monkeypatch, "outscore")
    window._on_score_clicked()

    result = window._score_result
    assert result["method"] == "outscore"
    assert result["constants"][0] == pytest.approx(slow, rel=0.03)
    assert result["constants"][1] == pytest.approx(fast, rel=0.03)
    assert "OUTSCORE" in window.score_results_label.text()


def test_main_window_score_dialog_offers_both_methods(qapp):
    dialog = _ScoreSettingsDialog(suggested=2)
    methods = [dialog.method_combo.itemData(i) for i in range(dialog.method_combo.count())]
    assert methods == ["score", "outscore"]
    assert dialog.values()["method"] == "score"  # the safe default


def test_main_window_score_dialog_forbids_one_component_for_outscore(qapp):
    # Cross-talk is defined between pairs.
    dialog = _ScoreSettingsDialog(suggested=1)
    assert dialog.components_spin.minimum() == 1
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("outscore"))
    assert dialog.components_spin.minimum() == 2
    assert dialog.components_spin.value() >= 2


def test_main_window_score_dialog_states_the_data_requirements(qapp):
    # The methods fail quietly, so what they need is said where the
    # choice is made rather than only in the README.
    dialog = _ScoreSettingsDialog(suggested=2)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "gradient levels" in text
    assert "baseline-correct" in text.lower() or "baseline" in text.lower()
    assert "10%" in text


def test_main_window_warns_when_the_resolution_may_be_unreliable(qapp, monkeypatch):
    # The overlapping-species window shares a resonance, which is the
    # case OUTSCORE's assumption does not cover.
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    _accept_score_dialog(monkeypatch, "outscore")
    warned = {}
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warned.setdefault("body", args[2]))
    window._on_score_clicked()
    assert "share resonances" in warned.get("body", "")
    # The result is still produced -- the warning informs, it does not block.
    assert window._score_result is not None


def test_main_window_export_buttons_hidden_until_a_resolution_exists(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    assert window.keep_components_button.isHidden() is True
    _accept_score_dialog(monkeypatch)
    window._on_score_clicked()
    assert window.keep_components_button.isHidden() is False
    assert window.export_components_button.isHidden() is False


def test_main_window_keeps_components_as_real_spectra(qapp, monkeypatch):
    window, _b, ppm = _overlapping_pfg_window(qapp)
    _accept_score_dialog(monkeypatch)
    window._on_score_clicked()
    before = len(window.session)

    window._on_keep_components_clicked()

    assert len(window.session) == before + 2
    kept = window.session.spectra[before]
    assert kept.ndim == 1
    np.testing.assert_allclose(kept.data, window._score_result["spectra"][0])
    np.testing.assert_allclose(kept.axes[0].values("ppm"), ppm)
    # Labelled by its own diffusion coefficient, not just a number.
    label = window.session.label(before)
    assert "component 1" in label and "m^2/s" in label
    # And carrying its provenance.
    assert kept.metadata.raw["score_component"] == 1
    assert kept.metadata.raw["score_constant"] == pytest.approx(
        window._score_result["constants"][0])


def test_main_window_exports_components_to_csv(qapp, monkeypatch, tmp_path):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    _accept_score_dialog(monkeypatch)
    window._on_score_clicked()

    path = tmp_path / "components.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args, **kwargs: (str(path), "CSV files (*.csv)"))
    window._on_export_components_clicked()

    text = path.read_text(encoding="utf-8")
    assert "SCORE" in text
    assert "component_1" in text and "component_2" in text
    assert "Decay constant" in text


def test_main_window_export_without_a_resolution_says_so(qapp, monkeypatch):
    window = MainWindow()
    seen = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: seen.setdefault("title", args[1]))
    window._on_export_components_clicked()
    assert seen.get("title") == "Nothing to export"
    window._on_keep_components_clicked()
    assert seen.get("title") in ("Nothing to export", "Nothing to keep")


# ============================================================
# Echo position nudge
# ============================================================

def test_main_window_echo_offset_only_shown_while_aligning(qapp, tmp_path, monkeypatch):
    # It corrects the *detected* echo top, and nothing is detecting one
    # unless alignment is on.
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    assert window.echo_offset_spin.isHidden() is True
    window.align_echo_button.setChecked(True)
    assert window.echo_offset_spin.isHidden() is False
    window.align_echo_button.setChecked(False)
    assert window.echo_offset_spin.isHidden() is True


def test_main_window_echo_offset_moves_the_marker(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window.align_echo_button.setChecked(True)

    def echo_marker(canvas):
        line = next(a for a in canvas.ax.lines if a.get_label() == "Echo top")
        return float(line.get_xdata()[0])

    window.echo_offset_spin.setValue(0.0)
    baseline = echo_marker(window.fid_canvas)
    window.echo_offset_spin.setValue(3.0)
    moved = echo_marker(window.fid_canvas)
    # The marker is drawn in the FID's own x units, so the shift is
    # three samples' worth rather than three units.
    assert moved > baseline


def test_main_window_echo_offset_round_trips_through_reselection(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="a"), vendor="bruker")
    window.align_echo_button.setChecked(True)
    window.echo_offset_spin.setValue(-2.5)
    window._persist_fid_processing_controls_to_metadata(window.session.spectra[0])

    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="b"), vendor="bruker")
    assert window.echo_offset_spin.value() == pytest.approx(0.0)   # b has none saved
    window._select_single_row(0)
    assert window.echo_offset_spin.value() == pytest.approx(-2.5)


def test_main_window_echo_offset_reaches_the_reader_kwargs(qapp):
    # It maps onto a real reader parameter, so a reopened spectrum is
    # read already offset rather than read and then recomputed.
    kwargs = MainWindow._saved_processing_read_kwargs(
        {"live_align_echo": True, "live_echo_offset": 1.5, "live_truncate": 900})
    assert kwargs["echo_offset"] == pytest.approx(1.5)
    assert kwargs["truncate"] == 900


# ============================================================
# SCORE: on-plot annotations, Analysis-tab entry, relative heights
# ============================================================

def test_main_window_score_annotates_components_on_the_plot(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()

    assert window.score_annotate_check.isChecked() is True
    texts = [t.get_text() for axes in window.fit_canvas._compare_axes for t in axes.texts]
    assert len(texts) == 2
    # Each carries the fitted value *and* the share -- the pair a
    # resolved component is quoted by.
    for text in texts:
        assert "m^2/s" in text
        assert "%" in text


def test_main_window_score_annotation_includes_a_ratio_to_the_largest(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    assert any("× the largest" in t.get_text()
               for axes in window.fit_canvas._compare_axes for t in axes.texts)


def test_main_window_score_annotations_can_be_turned_off(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    window.score_annotate_check.setChecked(False)
    assert all(not axes.texts for axes in window.fit_canvas._compare_axes)


def test_main_window_score_shows_true_relative_heights_by_default(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()

    assert window.score_shared_scale_check.isChecked() is True
    ranges = [abs(ax.get_ylim()[1] - ax.get_ylim()[0]) for ax in window.fit_canvas._compare_axes]
    assert ranges[0] == pytest.approx(ranges[1])

    # Off, each row is autoscaled to fill itself again.
    window.score_shared_scale_check.setChecked(False)
    ranges = [abs(ax.get_ylim()[1] - ax.get_ylim()[0]) for ax in window.fit_canvas._compare_axes]
    assert ranges[0] != pytest.approx(ranges[1])


def test_main_window_score_runs_from_the_analysis_tab(qapp, monkeypatch):
    # Explicit request: reachable from the Analysis tab, not only the
    # menu bar. Same handler either way.
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    assert window.resolve_components_button.isEnabled() is True

    window.resolve_components_button.click()
    assert window._score_result is not None


def test_main_window_score_display_toggles_hidden_until_a_result_exists(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    assert window.score_annotate_check.isHidden() is True
    assert window.score_shared_scale_check.isHidden() is True
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    assert window.score_annotate_check.isHidden() is False
    assert window.score_shared_scale_check.isHidden() is False


def test_main_window_score_components_can_switch_axis_units(qapp, monkeypatch):
    window, _b, ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()

    window.fit_canvas.set_axis_unit("khz")
    assert window.fit_canvas.axis_unit == "khz"
    expected = ppm * 400.0 / 1000.0
    for axes in window.fit_canvas._compare_axes:
        np.testing.assert_allclose(axes.lines[0].get_xdata(), expected)


# ============================================================
# X-Pulse phase persistence: applied before the first plot
# ============================================================

def _real_part_fraction(spectrum, row=0):
    """Real-part peak as a fraction of the magnitude peak -- 1.0 when
    perfectly phased, near 0 when the signal is still on the imaginary
    channel."""
    complex_row = spectrum.data[row] + 1j * spectrum.data_imag[row]
    magnitude = np.abs(complex_row).max()
    return float(np.real(complex_row).max() / magnitude) if magnitude else 0.0


def _residual_phase(spectrum, row, coarse=1.0):
    """How much more zero-order phase ``row`` still wants, in degrees.
    0 means correctly phased."""
    return processing_ft.apc0(spectrum.complex_data[row], spectrum.axis_values(-1, "ppm"), coarse)


def _pfg_workdir(tmp_path, name="4", source=("test_data", "xpulse_PFG", "2026.08.22", "4")):
    src = os.path.join(*source)
    if not os.path.isdir(src):
        pytest.skip(f"{src} not available")
    work = tmp_path / name
    shutil.copytree(src, work)
    return str(work)


def test_main_window_xpulse_phase_composes_with_its_saved_recipe(qapp, tmp_path):
    # The X-Pulse base comes from autophase, which returns a
    # pivot-model phase (exp(-i*phi)); the recorded slot it was stored
    # in means TopSpin's index ramp (exp(+i*phi)). Copied across
    # verbatim, the two cancelled on every rebuild from the raw FID,
    # so a spectrum with a saved sidecar reloaded essentially unphased
    # while the same data with no sidecar reloaded correctly. Reported
    # as phasing being reset "in and out of the processing view".
    window = MainWindow()
    window._load(_pfg_workdir(tmp_path), vendor="xpulse")
    spectrum = window.session.spectra[0]
    # Measured before the fix: ~-90 deg still wanted on every row.
    for row in range(spectrum.data.shape[0]):
        assert abs(_residual_phase(spectrum, row, coarse=0.5)) < 15.0


def test_main_window_load_reanchors_the_phase_reference_after_reprocessing(qapp, tmp_path):
    # Selecting the row captures a phase reference; applying the saved
    # recipe then replaces spectrum.data behind it at the *same shape*,
    # which _phase_reference_is_stale cannot see. The next phase tick
    # wrote that stale snapshot back over every row, undoing the
    # recompute -- measured at 9.3 deg of rotation and a 1.043 rescale
    # on this dataset, from touching a phase control and nothing else.
    window = MainWindow()
    window._load(_pfg_workdir(tmp_path), vendor="xpulse")
    spectrum = window.session.spectra[0]
    loaded = spectrum.complex_data.copy()

    window.apply_all_slices_checkbox.setChecked(False)
    window.slice_spin.setValue(3)
    window._apply_phase_preview()  # a zero-delta tick: must be a no-op

    np.testing.assert_allclose(spectrum.complex_data, loaded)


def test_main_window_pseudo2d_slice_phase_survives_a_slice_change(qapp, tmp_path):
    # A per-slice phase used to live only in the spin boxes, which a
    # slice change zeroes -- so it read as "reset" the moment you
    # looked at another slice and came back.
    window = MainWindow()
    window._load(_pfg_workdir(tmp_path), vendor="xpulse")
    spectrum = window.session.spectra[0]

    window.apply_all_slices_checkbox.setChecked(False)
    window.slice_spin.setValue(3)
    window.phase0_spin.setValue(45.0)
    window._apply_phase_preview()
    phased_row = spectrum.complex_data[2].copy()
    shown = window.phase0_current_display.text()

    window.slice_spin.setValue(5)
    window.slice_spin.setValue(3)

    assert spectrum.metadata.raw["slice_phase0"][2] == pytest.approx(45.0)
    # Only that row -- a per-slice phase is not a dataset-wide one.
    assert [v for i, v in enumerate(spectrum.metadata.raw["slice_phase0"]) if i != 2] == \
        [0.0] * (spectrum.data.shape[0] - 1)
    np.testing.assert_allclose(spectrum.complex_data[2], phased_row)
    assert window.phase0_current_display.text() == shown


def test_main_window_pseudo2d_slice_phase_survives_a_reprocess(qapp, tmp_path):
    # The rebuild-from-raw that every live processing control and every
    # settled periodic save runs used to apply only the dataset-wide
    # phase, silently returning individually-phased slices to it.
    window = MainWindow()
    window._load(_pfg_workdir(tmp_path), vendor="xpulse")
    spectrum = window.session.spectra[0]

    window.apply_all_slices_checkbox.setChecked(False)
    window.slice_spin.setValue(3)
    window.phase0_spin.setValue(45.0)
    window._apply_phase_preview()
    phased = spectrum.complex_data.copy()

    window._capture_phase_reference()  # what a slice/tab change does: commit, then re-zero
    window._recompute_multi_row_data(spectrum)

    # Exactly, not approximately: the rebuild composes the same
    # rotations from the same raw FID.
    np.testing.assert_allclose(spectrum.complex_data, phased)


def test_main_window_processing_preview_matches_a_per_slice_phased_row(qapp, tmp_path):
    # "It seems like the pseudo-2D phase is not what I see in the
    # processing window": that preview rebuilds from the raw FID plus
    # metadata.raw, and per-slice phase was in neither.
    window = MainWindow()
    window._load(_pfg_workdir(tmp_path), vendor="xpulse")
    spectrum = window.session.spectra[0]

    window.apply_all_slices_checkbox.setChecked(False)
    window.slice_spin.setValue(3)
    window.phase0_spin.setValue(45.0)
    window._apply_phase_preview()
    window._capture_phase_reference()  # commits the drag and re-zeros the boxes

    preview = window._compute_processing_preview(spectrum)[2]
    spectrum_row = spectrum.complex_data[2]
    peak = int(np.argmax(np.abs(spectrum_row)))
    window_slice = slice(peak - 50, peak + 50)
    ratio = np.mean(preview.complex_data[window_slice] / spectrum_row[window_slice])
    assert float(np.rad2deg(np.angle(ratio))) == pytest.approx(0.0, abs=0.5)


def test_main_window_autophase_each_slice_records_what_it_applied(qapp, tmp_path):
    # Applying a per-row phase without recording it meant the next
    # rebuild threw it away.
    window = MainWindow()
    window._load(_pfg_workdir(tmp_path), vendor="xpulse")
    spectrum = window.session.spectra[0]
    window.apply_all_slices_checkbox.setChecked(True)
    window._on_autophase_clicked()

    recorded = spectrum.metadata.raw["slice_phase0"]
    assert len(recorded) == spectrum.data.shape[0]
    assert any(recorded)
    phased = spectrum.complex_data.copy()
    window._recompute_multi_row_data(spectrum)
    np.testing.assert_allclose(spectrum.complex_data, phased)


def test_main_window_xpulse_pfg_is_phased_on_load_without_interaction(qapp):
    # Reported directly: phase persistence "wasn't working on the PFG
    # experiment", and other pseudo-2Ds needed "jump starting" by
    # touching a phase control before the plot caught up.
    #
    # Both were one cause: X-Pulse records no phase of its own, so the
    # base comes from autophase at read time -- and that was stored in
    # the *user* slot, where the saved sidecar's own phase0 overwrote
    # it. The rebuild-from-raw then applied the user's correction with
    # nothing under it. Measured before the fix: 0.118, i.e. the signal
    # essentially all still on the imaginary channel.
    path = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    window = MainWindow()
    window._load(path, vendor="xpulse")
    spectrum = window.session.spectra[0]

    # No user interaction at all -- this is the state the first plot
    # is drawn from.
    assert _real_part_fraction(spectrum) > 0.5
    # And the base is where a rebuild will find it.
    assert float(spectrum.metadata.raw.get("recorded_phase0", 0.0)) != 0.0


def test_main_window_xpulse_phase_survives_a_save_and_reload(qapp, tmp_path):
    # The round trip that "persistence" actually means: whatever phase
    # the spectrum is left in is exactly what comes back, with nothing
    # touched in between.
    source = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(source):
        pytest.skip(f"{source} not available")
    work = tmp_path / "pfg"
    shutil.copytree(source, work)

    window = MainWindow()
    window._load(str(work), vendor="xpulse")
    spectrum = window.session.spectra[0]
    window.apply_all_slices_checkbox.setChecked(True)
    window._on_autophase_clicked()
    window._commit_pending_phase_to_metadata()
    tuned = _real_part_fraction(spectrum)
    io_persistence.save_processing_sidecar(spectrum)

    reopened = MainWindow()
    reopened._load(str(work), vendor="xpulse")
    # Not bit-exact, and it should not be claimed to be: the in-session
    # spectrum was phased by applying a delta to already-phased data,
    # while the reload rebuilds from the raw FID and applies
    # recorded-plus-user in one step. Different arithmetic, same
    # answer to ~3e-5 -- orders of magnitude below anything visible or
    # scientifically meaningful, and the point being tested is that
    # the phase comes back *at all*, which it previously did not
    # (0.118 against 0.73).
    assert _real_part_fraction(reopened.session.spectra[0]) == pytest.approx(tuned, abs=1e-3)


def test_main_window_defers_the_first_draw_when_a_recipe_will_change_the_data(qapp, tmp_path):
    # The other half of "factored in before the first plotting": with a
    # saved recipe, the initial draw is deferred until after the
    # recompute, so the first thing rendered is the corrected data
    # rather than the vendor default being drawn and then replaced.
    source = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(source):
        pytest.skip(f"{source} not available")
    work = tmp_path / "pfg"
    shutil.copytree(source, work)

    window = MainWindow()
    drawn = []
    real_draw = MainWindow._draw_selected_spectrum

    def spy(self, spectrum, *args, **kwargs):
        drawn.append(_real_part_fraction(spectrum))
        return real_draw(self, spectrum, *args, **kwargs)

    MainWindow._draw_selected_spectrum = spy
    try:
        window._load(str(work), vendor="xpulse")
    finally:
        MainWindow._draw_selected_spectrum = real_draw

    assert drawn, "the spectrum was never drawn"
    # Every draw that happened saw phased data -- there is no
    # intermediate unphased render to 'catch up' from.
    assert all(fraction > 0.5 for fraction in drawn)


# ============================================================
# Autophasing a pseudo-2D whose slices disagree
# ============================================================

def _drifting_phase_series(n_rows=6, drift_deg=120.0, npts=512):
    """A pseudo-2D whose rows are identical except for a phase that
    drifts across the series -- what an X-Pulse PFG stimulated-echo
    experiment actually produces."""
    ppm = np.linspace(0.0, 10.0, npts)
    line = 1.0 / (1.0 + ((ppm - 5.0) / 0.05) ** 2)
    rows_r, rows_i = [], []
    for k in range(n_rows):
        phase = np.deg2rad(drift_deg * k / max(n_rows - 1, 1))
        z = line * np.exp(1j * phase)
        rows_r.append(np.real(z))
        rows_i.append(np.imag(z))
    return NMRSpectrum(
        data=np.array(rows_r), data_imag=np.array(rows_i),
        axes=[NMRAxis(kind="index", index=np.arange(1, n_rows + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="xpulse", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )


def _real_fraction(spectrum, row):
    z = spectrum.data[row] + 1j * spectrum.data_imag[row]
    return float(np.real(z).max() / np.abs(z).max())


def test_main_window_autophase_all_slices_phases_every_row(qapp):
    # There is no shared-phase compromise left to get wrong: with
    # "Apply to every slice" ticked, Autophase gives each row its own
    # phase0. Optimising one shared phase against the stacked block
    # used to take slice 1 of a real X-Pulse PFG from 0.99 to 0.73.
    spectrum = _drifting_phase_series()
    # One row unambiguously the strongest -- which used to decide the
    # single phase every other row then had to live with.
    spectrum.data[2] *= 5.0
    spectrum.data_imag[2] *= 5.0

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.apply_all_slices_checkbox.setChecked(True)
    window._on_autophase_clicked()

    result = window.session.spectra[0]
    for row in range(result.data.shape[0]):
        assert _real_fraction(result, row) > 0.98


def test_main_window_warns_when_slices_disagree_about_phase(qapp):
    spectrum = _drifting_phase_series(drift_deg=120.0)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    # Phasing *this* slice only -- which is right for it and says
    # nothing about the rest.
    window.apply_all_slices_checkbox.setChecked(False)

    window._on_autophase_clicked()

    # Reported without blocking: autophase is a one-keystroke action a
    # user may repeat while trying settings, and the status label sits
    # right beside the phase controls.
    message = window.processing_status_label.text()
    assert "disagree about their phase" in message
    # And it names the control that can actually do the job.
    assert "Apply to every slice" in message


def test_main_window_does_not_warn_when_slices_agree(qapp):
    spectrum = _drifting_phase_series(drift_deg=0.0)   # every row the same phase
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.apply_all_slices_checkbox.setChecked(False)

    window._on_autophase_clicked()
    assert "disagree about their phase" not in window.processing_status_label.text()


def test_phase_spread_is_circular(qapp):
    # Rows at -179 and +179 are two degrees apart, not 358.
    ppm = np.linspace(0.0, 10.0, 256)
    line = 1.0 / (1.0 + ((ppm - 5.0) / 0.05) ** 2)
    block = np.array([line * np.exp(1j * np.deg2rad(angle)) for angle in (-179.0, 179.0)])
    window = MainWindow()
    spread = window._pseudo2d_phase_spread(block, ppm)
    assert spread is not None and spread < 10.0


def test_phase_spread_ignores_near_noise_rows(qapp):
    # A near-noise row's own "best" phase is arbitrary and would
    # manufacture a disagreement that is not there.
    ppm = np.linspace(0.0, 10.0, 256)
    line = 1.0 / (1.0 + ((ppm - 5.0) / 0.05) ** 2)
    strong = [line * np.exp(1j * np.deg2rad(a)) for a in (10.0, 11.0, 12.0)]
    faint = [1e-6 * line * np.exp(1j * np.deg2rad(a)) for a in (-170.0, 150.0)]
    window = MainWindow()
    spread = window._pseudo2d_phase_spread(np.array(strong + faint), ppm)
    assert spread is not None and spread < window._PSEUDO2D_PHASE_SPREAD_LIMIT


def test_main_window_per_slice_autophase_fixes_a_drifting_series(qapp):
    # The control the warning points at must actually work.
    spectrum = _drifting_phase_series(drift_deg=150.0)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.apply_all_slices_checkbox.setChecked(True)
    window._on_autophase_clicked()
    result = window.session.spectra[0]
    for row in range(result.data.shape[0]):
        assert _real_fraction(result, row) > 0.98


# ============================================================
# PFG: ramp, reading timings from file, probe gradient store
# ============================================================

def test_main_window_b_from_gradients_opens_no_stray_window(qapp):
    # Reported directly: "a (sometimes tiny) blank popup box appears
    # when the 'b from gradients' item is selected". One entry in
    # pfg_widgets paired the Save-gradient button with a brand-new
    # QLabel("") that was never added to any layout -- and
    # setVisible(True) on a parentless widget makes Qt show it as its
    # own top-level window.
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)

    for label, widget in window.pfg_widgets:
        for part in (label, widget):
            assert part.parentWidget() is not None, (
                f"{part!r} has no parent, so showing it opens a window of its own")


def test_main_window_warns_when_no_value_list_was_found(qapp):
    # Reported directly: "there should be some kind of warning if the
    # values are not found from the file". Auto quietly fell back to
    # slice indices, and a T2 fitted against 1, 2, 3, ... reports a
    # per-slice rate that looks exactly like a real one.
    delays = np.geomspace(0.001, 0.3, 8)
    spectrum = _make_pseudo2d_relaxation_spectrum(delays, x0=5.0, fwhm=0.05, T2=0.05)
    spectrum.metadata.source_path = None
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(0)      # Auto
    window._refresh_values_preview()

    preview = window.values_preview_label.text()
    assert "⚠" in preview
    assert "no value list found" in preview

    # And it travels with the result, which is where the number that
    # gets written down is read from.
    window._on_fit_curve_button_clicked()
    assert "no value list found" in window.curve_fit_summary_label.text()


def test_main_window_does_not_warn_when_the_values_are_real(qapp):
    delays = np.geomspace(0.001, 0.3, 8)
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    assert "⚠" not in window.values_preview_label.text()
    assert "no value list found" not in window.analysis_results_label.text()


def test_main_window_warns_when_no_gradient_list_was_found(qapp):
    # The other silent assumption: with no recorded gradient list,
    # b-from-gradients invents an even 0->100% ramp.
    b = np.linspace(0.0, 3.0e9, 8)
    ppm = np.linspace(0.0, 10.0, 128)
    data = np.array([lorentzian(ppm, 5.0, 0.08, 1.0) * np.exp(-1.0e-9 * bi) for bi in b])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, b.size + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)
    window.pfg_max_gradient_spin.setValue(50.0)
    window.pfg_delta_spin.setValue(1.0)
    window.pfg_big_delta_spin.setValue(50.0)
    window._refresh_values_preview()

    preview = window.values_preview_label.text()
    assert "no gradient list found" in preview


def test_main_window_ramp_box_only_shown_for_the_trapezoid(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)
    window.pfg_shape_combo.setCurrentIndex(window.pfg_shape_combo.findData("smsq10"))
    assert window.pfg_ramp_spin.isHidden() is True
    window.pfg_shape_combo.setCurrentIndex(window.pfg_shape_combo.findData("trapezoid"))
    assert window.pfg_ramp_spin.isHidden() is False


def test_main_window_ramp_reaches_the_b_value_calculation(qapp):
    from opennmr.processing.fitting import lorentzian

    relative = np.linspace(0.1, 1.0, 6)
    ppm = np.linspace(0.0, 10.0, 128)
    # Real data, not zeros: an all-zero spectrum cannot be plotted, and
    # _draw_selected_spectrum reports that through a modal dialog --
    # which hangs a headless test rather than failing it.
    data = np.array([lorentzian(ppm, 5.0, 0.1, 1.0 - 0.1 * k) for k in range(6)])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, 7)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="xpulse", nucleus="1H", sfo1=400.0, pseudo_2d=True,
                              raw={"gradient_relative": relative}),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)
    window.pfg_shape_combo.setCurrentIndex(window.pfg_shape_combo.findData("trapezoid"))
    window.pfg_max_gradient_spin.setValue(50.0)

    window.pfg_ramp_spin.setValue(0.0)
    plain = window._resolve_independent_values(spectrum, is_diffusion=True, n_slices=6)
    window.pfg_ramp_spin.setValue(0.4)
    ramped = window._resolve_independent_values(spectrum, is_diffusion=True, n_slices=6)
    assert np.all(ramped[1:] < plain[1:])


def test_main_window_reads_pfg_timings_from_a_real_acquisition_file(qapp):
    path = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    window = MainWindow()
    window._load(path, vendor="xpulse")
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(window._B_FROM_GRADIENTS_SOURCE)

    window._on_read_pfg_params_clicked()

    # delta is the plateau (4 ms) plus one ramp (0.5 ms) -- the file
    # records the plateau alone, the b-value expression wants the sum.
    assert window.pfg_delta_spin.value() == pytest.approx(4.5)
    assert window.pfg_big_delta_spin.value() == pytest.approx(50.0)
    assert window.pfg_ramp_spin.value() == pytest.approx(0.5)
    # A recorded ramp *is* the statement that the gradients are ramped.
    assert window.pfg_shape_combo.currentData() == "trapezoid"
    # The message gained per-value source tags when the Bruker/Spinsolve
    # readers were wired in, so it names the experiment's files rather
    # than "the acquisition file" -- assert on what it now promises:
    # the values, and where each one came from.
    message = window.statusBar().currentMessage()
    assert "Read from the experiment's files" in message
    assert "[xpulse acquisition parameters]" in message


def test_main_window_says_so_when_no_timings_are_recorded(qapp, monkeypatch, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=32), vendor="bruker")
    seen = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: seen.setdefault("title", args[1]))
    window._on_read_pfg_params_clicked()
    assert seen.get("title") == "Nothing recorded"


def test_main_window_pfg_and_imaging_share_one_probe_gradient_store(qapp, tmp_path, monkeypatch):
    # It is one physical property of the probe -- setting it in either
    # place must set it for both.
    config_path = tmp_path / "probe_config.json"
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    spectrum = _make_1d_peak_spectrum(x0=5.0, fwhm=0.05)
    spectrum.metadata.probe = "diffBB"
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)

    window.pfg_max_gradient_spin.setValue(1702.41)
    window._save_probe_max_gradient(window.pfg_max_gradient_spin.value())

    saved = json.loads(config_path.read_text())
    assert saved["diffBB"]["max_gradient_g_per_cm"] == {"z": pytest.approx(1702.41)}
    # And the Imaging tab's own button writes the same key -- and its
    # own box already shows what was typed in the PFG one, since the
    # maximum gradient is one per-experiment value, not one per tab.
    assert window.imaging_max_gradient_spin.value() == pytest.approx(1702.41)
    window.imaging_max_gradient_spin.setValue(2897.61)
    window._on_save_probe_gradient_default_clicked()
    assert json.loads(config_path.read_text())["diffBB"]["max_gradient_g_per_cm"] == \
        {"z": pytest.approx(2897.61)}
    assert window.pfg_max_gradient_spin.value() == pytest.approx(2897.61)


def test_main_window_probe_gradient_is_saved_per_axis(qapp, tmp_path, monkeypatch):
    # A triple-axis probe's three coils are rarely the same strength,
    # and one shared number silently scales every D (or mm axis)
    # measured on the other two.
    config_path = tmp_path / "probe_config.json"
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    spectrum = _make_1d_peak_spectrum(x0=5.0, fwhm=0.05)
    spectrum.metadata.probe = "triple"
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)

    window.pfg_gradient_axis_combo.setCurrentIndex(window.pfg_gradient_axis_combo.findData("z"))
    window.pfg_max_gradient_spin.setValue(300.0)
    window._save_probe_max_gradient(300.0)
    window.pfg_gradient_axis_combo.setCurrentIndex(window.pfg_gradient_axis_combo.findData("x"))
    window.pfg_max_gradient_spin.setValue(55.0)
    window._save_probe_max_gradient(55.0)

    saved = json.loads(config_path.read_text())["triple"]["max_gradient_g_per_cm"]
    # Saving x left z alone -- one axis at a time, not a replacement.
    assert saved == {"z": pytest.approx(300.0), "x": pytest.approx(55.0)}


def test_main_window_switching_gradient_axis_swaps_in_that_axis_s_saved_value(qapp, tmp_path, monkeypatch):
    config_path = tmp_path / "probe_config.json"
    config_path.write_text(json.dumps({"triple": {"max_gradient_g_per_cm": {"z": 300.0, "y": 61.0}}}))
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(config_path))

    spectrum = _make_1d_peak_spectrum(x0=5.0, fwhm=0.05)
    spectrum.metadata.probe = "triple"
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window.pfg_max_gradient_spin.value() == pytest.approx(300.0)  # z, the default axis

    window.imaging_gradient_axis_combo.setCurrentIndex(window.imaging_gradient_axis_combo.findData("y"))
    # Both combos follow, and the number follows the axis -- leaving z's
    # 300 in force while the axis says y is exactly the silent error
    # the per-axis store exists to prevent.
    assert window.pfg_gradient_axis_combo.currentData() == "y"
    assert window.pfg_max_gradient_spin.value() == pytest.approx(61.0)
    assert window.imaging_max_gradient_spin.value() == pytest.approx(61.0)


def test_main_window_probe_gradient_reads_the_pre_per_axis_format(qapp, tmp_path, monkeypatch):
    # A config written before the store was per-axis has a bare number
    # and no axis. It could not say which coil it described, so it
    # answers for any of them rather than being discarded.
    config_path = tmp_path / "probe_config.json"
    config_path.write_text(json.dumps({"legacyprobe": {"max_gradient_g_per_cm": 42.5}}))
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(config_path))

    window = MainWindow()
    assert window._probe_axis_gradients("legacyprobe") == {
        "z": pytest.approx(42.5), "x": pytest.approx(42.5), "y": pytest.approx(42.5),
    }


def test_main_window_pfg_gradient_prefills_from_the_probe_default(qapp, tmp_path, monkeypatch):
    config_path = tmp_path / "probe_config.json"
    config_path.write_text(json.dumps({"diffBB": {"max_gradient_g_per_cm": 1702.41}}))
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(config_path))

    spectrum = _make_1d_peak_spectrum(x0=5.0, fwhm=0.05)
    spectrum.metadata.probe = "diffBB"
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window.pfg_max_gradient_spin.value() == pytest.approx(1702.41)


def test_main_window_probe_default_does_not_overwrite_this_spectrum_s_own_value(qapp, tmp_path, monkeypatch):
    # A probe default is a starting point: it fills in for an
    # experiment that has no value of its own, and never replaces one
    # that does. A number typed into either box *is* this experiment's
    # own value (see MainWindow._on_max_gradient_changed), so it
    # outranks the probe default from then on.
    config_path = tmp_path / "probe_config.json"
    config_path.write_text(json.dumps({"diffBB": {"max_gradient_g_per_cm": {"z": 1702.41}}}))
    monkeypatch.setattr(MainWindow, "_PROBE_CONFIG_PATH", str(config_path))

    window = MainWindow()
    spectrum = _make_1d_peak_spectrum(x0=5.0, fwhm=0.05)
    spectrum.metadata.probe = "diffBB"
    _load_spectrum_directly(window, spectrum)
    assert window.pfg_max_gradient_spin.value() == pytest.approx(1702.41)  # probe default fills in

    window.pfg_max_gradient_spin.setValue(37.0)
    window._refresh_max_gradient_controls(window.session.spectra[0])
    assert window.pfg_max_gradient_spin.value() == pytest.approx(37.0)
    assert window.imaging_max_gradient_spin.value() == pytest.approx(37.0)


# ============================================================
# The processing view survives slice changes and settles
# ============================================================

def _preview_limits(window):
    return window.processing_preview_canvas.processing_preview_limits()


def test_main_window_processing_view_survives_a_slice_change(qapp, tmp_path):
    # Explicit user request: comparing one slice against the next at a
    # fixed zoom is the reason for stepping through them at all.
    # Real data: the dual FID+spectrum preview only exists for a
    # spectrum that can be reprocessed from its raw FID, which a
    # synthetic in-memory 2D cannot.
    path = os.path.join("test_data", "bruker_T2", "11")
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    window._show_processing_preview(window.session.spectra[0], 0)
    assert window.processing_preview_canvas.processing_preview_limits() is not None

    window.processing_preview_canvas.figure.axes[1].set_xlim(4.0, 6.0)
    window.processing_preview_canvas.figure.axes[1].set_ylim(0.1, 0.2)

    window.slice_spin.setValue(3)

    # Re-read the axes rather than holding a reference: the rebuild
    # goes through _clear_figure, which destroys the old Axes objects.
    spec_ax = window.processing_preview_canvas.figure.axes[1]
    assert spec_ax.get_xlim() == pytest.approx((4.0, 6.0))
    assert spec_ax.get_ylim() == pytest.approx((0.1, 0.2))


def test_main_window_processing_view_survives_a_settled_reprocess(qapp, tmp_path):
    # The "periodic saves" case: the debounced rebuild fires after a
    # pause and used to snap the zoom back out.
    path = os.path.join("test_data", "bruker_T2", "11")
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    window._show_processing_preview(window.session.spectra[0], 0)

    window.processing_preview_canvas.figure.axes[1].set_xlim(3.0, 7.0)
    before = window.processing_preview_canvas.figure.axes[1].get_xlim()

    window._refresh_processing_preview_if_active(preserve_view=True)
    assert window.processing_preview_canvas.figure.axes[1].get_xlim() == pytest.approx(before)


def test_processing_preview_limits_round_trip(qapp, tmp_path):
    path = os.path.join("test_data", "bruker_T2", "11")
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    window._show_processing_preview(window.session.spectra[0], 0)

    captured = _preview_limits(window)
    assert captured is not None and len(captured) == 2
    window.processing_preview_canvas.figure.axes[1].set_xlim(1.0, 2.0)
    window.processing_preview_canvas.restore_processing_preview_limits(captured)
    assert window.processing_preview_canvas.figure.axes[1].get_xlim() == \
        pytest.approx(captured[1][0])


# ============================================================
# DOSY behaves like a contour
# ============================================================

def _dosy_window(qapp, monkeypatch):
    from opennmr.processing.fitting import lorentzian

    b = np.linspace(0.0, 4.0e9, 24)
    ppm = np.linspace(0.0, 10.0, 300)
    data = np.zeros((b.size, ppm.size))
    for i, bi in enumerate(b):
        data[i] += lorentzian(ppm, 3.0, 0.08, 1.0) * np.exp(-1.5e-9 * bi)
        data[i] += lorentzian(ppm, 7.0, 0.08, 1.0) * np.exp(-1.2e-10 * bi)
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, b.size + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{v:.8g}" for v in b))
    monkeypatch.setattr(_DosySettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_dosy_clicked()
    window.fit_canvas.figure.set_size_inches(8, 6)
    window.fit_canvas.canvas.draw()
    return window


def test_dosy_is_drawn_as_an_unfilled_full_frame_contour(qapp, monkeypatch):
    # Reported directly: the DOSY "looks bad aesthetically... should be
    # plotted as a normal contour plot (not filled) and match the
    # styles accordingly", with a full frame and no floating ticks
    # above the map.
    from matplotlib.collections import PathCollection

    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    ax = canvas.ax

    # A filled contour leaves PathCollections behind; line contours do not.
    assert not [c for c in ax.collections if isinstance(c, PathCollection)]
    assert ax.collections, "nothing was contoured at all"

    # Full frame, unlike this package's open-sided line plots.
    assert all(ax.spines[side].get_visible() for side in ("top", "right", "bottom", "left"))
    assert ax.get_xlabel() and ax.get_ylabel()
    # Font size is asserted at the library level instead (see
    # tests/test_plotting_fit.py) -- _apply_layout shrinks text to fit
    # the canvas, so what lands here depends on the widget's size.

    # The strip above it carries no ticks on either axis -- they had
    # nothing to belong to, since it has no spines either.
    projection_ax = canvas._dosy_projection_ax
    assert not any(spine.get_visible() for spine in projection_ax.spines.values())
    assert all(not tick.tick1line.get_visible() and not tick.tick2line.get_visible()
               for tick in projection_ax.xaxis.get_major_ticks())
    assert all(not tick.tick1line.get_visible() and not tick.tick2line.get_visible()
               for tick in projection_ax.yaxis.get_major_ticks())


def test_dosy_d_axis_fits_the_populated_range(qapp, monkeypatch):
    # The ILT grid spans several decades more than any real sample
    # occupies -- it has to, since where the populations lie is what is
    # being solved for. Plotting all of it squashed every peak into a
    # corner of a blank panel.
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    grid = canvas._dosy_result["D"]
    low, high = canvas.ax.get_ylim()

    assert low >= float(np.min(grid)) and high <= float(np.max(grid))
    assert (high / low) < (float(np.max(grid)) / float(np.min(grid)))


def test_dosy_scroll_scales_the_contour_levels(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    assert canvas._dosy_result is not None
    before = canvas._dosy_factor

    def body_scroll(button):
        # Rebuilt each time against the *current* Axes: adjusting the
        # levels redraws, which replaces them, and an event still
        # pointing at the old ones is silently ignored.
        ax = canvas.ax
        bbox = ax.get_window_extent()
        event = types.SimpleNamespace(
            name="scroll_event", canvas=canvas.canvas, inaxes=ax,
            x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 + 0.5 * bbox.height,
            step=1, button=button)
        event.xdata, event.ydata = ax.transData.inverted().transform((event.x, event.y))
        canvas._on_scroll(event)

    body_scroll("up")
    assert canvas._dosy_factor > before          # up raises the floor
    body_scroll("down")
    body_scroll("down")
    assert canvas._dosy_factor < before          # and down lowers it


def test_main_window_can_get_back_to_a_single_dosy_map(qapp, monkeypatch):
    # Reported directly: "still no way to get back to the previous DOSY
    # ILT, only the controls to change the plot." The list of
    # inversions was hidden below two entries, and the Fit canvas is
    # shared -- so running a curve fit left the one DOSY unreachable.
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    assert len(spectrum.metadata.raw["dosy_variants"]) == 1
    assert window.dosy_variant_group.isVisible() or not window.isVisible()

    # Something else takes over the Fit canvas.
    window.fit_canvas.clear()
    assert window.fit_canvas._dosy_result is None

    # Ticking the inversion brings it back, without re-inverting.
    called = []
    monkeypatch.setattr(processing_ilt, "dosy_map", lambda *a, **k: called.append(1))
    window.dosy_variant_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.dosy_variant_list.item(0).setCheckState(Qt.CheckState.Checked)

    assert called == []
    assert window.fit_canvas._dosy_result is spectrum.metadata.raw["dosy_variants"][0]["result"]
    assert window.fit_canvas.ax.collections


def test_main_window_keeps_every_dosy_inversion_of_a_spectrum(qapp, monkeypatch):
    # Explicit user request: "so multiple fitting modes can be trialled
    # against one another". A second run with different settings adds
    # an attempt rather than replacing the first.
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    assert len(spectrum.metadata.raw["dosy_variants"]) == 1

    window.dosy_settings_for_test = None
    monkeypatch.setattr(_DosySettingsDialog, "values", lambda self: {
        "auto_alpha": False, "alpha": 5e-2, "d_min": 1e-13, "d_max": 1e-8,
        "n_grid": 64, "threshold": 0.05,
    })
    window._on_dosy_clicked()

    variants = spectrum.metadata.raw["dosy_variants"]
    assert len(variants) == 2
    # Each is named by what makes it different, not "DOSY 1/2".
    assert all("α=" in v["label"] and "thr" in v["label"] for v in variants)
    assert variants[0]["label"] != variants[1]["label"]
    assert window.dosy_variant_group.isVisible() or not window.isVisible()


def test_main_window_repeating_dosy_settings_replaces_that_attempt(qapp, monkeypatch):
    # Re-running the same thing is a retry, not a second opinion.
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window._on_dosy_clicked()
    assert len(spectrum.metadata.raw["dosy_variants"]) == 1


def test_main_window_ticking_two_inversions_overlays_them(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    monkeypatch.setattr(_DosySettingsDialog, "values", lambda self: {
        "auto_alpha": False, "alpha": 5e-2, "d_min": 1e-13, "d_max": 1e-8,
        "n_grid": 64, "threshold": 0.05,
    })
    window._on_dosy_clicked()
    assert len(spectrum.metadata.raw["dosy_variants"]) == 2

    for row in range(window.dosy_variant_list.count()):
        window.dosy_variant_list.item(row).setCheckState(Qt.CheckState.Checked)

    canvas = window.fit_canvas
    assert canvas._dosy_overlay is not None and len(canvas._dosy_overlay) == 2
    assert len(canvas.ax.collections) == 2
    # Labelled by settings, so the legend says which is which.
    assert len(canvas.ax.get_legend().get_texts()) == 2


def test_main_window_keep_only_the_latest_dosy_inversion(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    monkeypatch.setattr(_DosySettingsDialog, "values", lambda self: {
        "auto_alpha": False, "alpha": 5e-2, "d_min": 1e-13, "d_max": 1e-8,
        "n_grid": 64, "threshold": 0.05,
    })
    window._on_dosy_clicked()
    assert len(spectrum.metadata.raw["dosy_variants"]) == 2

    window._on_clear_dosy_variants_clicked()

    remaining = spectrum.metadata.raw["dosy_variants"]
    assert len(remaining) == 1
    assert remaining[0]["result"] is spectrum.metadata.raw["dosy_result"]


def test_main_window_keeps_the_dosy_map_across_a_reselection(qapp, monkeypatch, tmp_path):
    # Reported directly: "DOSY plots are not being kept anywhere
    # visible and it seems like the only way to get them back is to
    # recompute them."
    window = _dosy_window(qapp, monkeypatch)
    computed = window.fit_canvas._dosy_result
    assert computed is not None

    window._load(_make_1d_dataset(tmp_path, name="elsewhere"), vendor="bruker")
    assert window.fit_canvas._dosy_result is None      # correctly cleared for the new spectrum

    window._select_single_row(0)

    # Back, and the *same* map object -- reused, not re-inverted.
    assert window.fit_canvas._dosy_result is computed
    assert window.fit_canvas.ax.collections, "the map should be drawn again"
    assert window.dosy_display_box.isVisible() or not window.isVisible()


def test_main_window_restoring_a_dosy_does_not_recompute_it(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]

    called = []
    monkeypatch.setattr(processing_ilt, "dosy_map",
                        lambda *a, **k: called.append(1))
    window._restore_dosy_for(spectrum)

    assert called == []
    assert window.fit_canvas._dosy_result is spectrum.metadata.raw["dosy_result"]


def test_main_window_overlays_several_dosy_maps(qapp, monkeypatch):
    # Explicit user request: "I'd also like to be able to overlay DOSY
    # plot ILT results".
    window = _dosy_window(qapp, monkeypatch)
    first = window.session.spectra[0]
    assert first.metadata.raw.get("dosy_result") is not None

    # A second spectrum carrying its own already-computed map.
    second = dataclasses.replace(
        first, metadata=dataclasses.replace(first.metadata, raw=dict(first.metadata.raw)))
    _load_spectrum_directly(window, second)
    _select_rows(window, [0, 1])

    window._on_overlay_dosy_clicked()

    canvas = window.fit_canvas
    assert canvas._dosy_overlay is not None and len(canvas._dosy_overlay) == 2
    assert len(canvas.ax.collections) == 2          # one contour set per map
    assert len(canvas.ax.get_legend().get_texts()) == 2
    # Extraction still has one unambiguous map to work from.
    assert canvas._dosy_result is canvas._dosy_overlay[0]


def test_main_window_overlay_dosy_needs_two_maps(qapp, monkeypatch, tmp_path):
    window = _dosy_window(qapp, monkeypatch)
    shown = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown.setdefault("msg", a[2]))
    _select_rows(window, [0])
    window._on_overlay_dosy_clicked()
    assert "two or more" in shown["msg"]
    assert window.fit_canvas._dosy_overlay is None


def test_main_window_dosy_overlay_colours_and_names_are_editable(qapp, monkeypatch):
    # Explicit user request: "I need to be able to also adjust the DOSY
    # overlay colours and legend names." Reached through the same
    # series interface every other multi-series plot answers, so this
    # needed no DOSY-specific control of its own.
    window = _dosy_window(qapp, monkeypatch)
    first = window.session.spectra[0]
    second = dataclasses.replace(
        first, metadata=dataclasses.replace(first.metadata, raw=dict(first.metadata.raw)))
    _load_spectrum_directly(window, second)
    _select_rows(window, [0, 1])
    window._on_overlay_dosy_clicked()

    canvas = window.fit_canvas
    entries = canvas.series()
    assert len(entries) == 2
    assert all(e.can_rename for e in entries)
    assert entries[0].colour != entries[1].colour

    canvas.set_series_style(entries[0].key, colour="#aa3377", label="sample A")

    assert canvas.series()[0].colour == "#aa3377"
    assert "sample A" in [t.get_text() for t in canvas.ax.get_legend().get_texts()]
    # And it survives the redraw a level change causes.
    window.dosy_levels_spin.setValue(11)
    assert canvas.series()[0].colour == "#aa3377"
    assert canvas.series()[0].label == "sample A"


def test_main_window_series_section_serves_every_plot_type(qapp, monkeypatch, tmp_path):
    # One dialog section, driven by MplCanvas.series() -- it used to be
    # an extracted-slices-only section, which is why the DOSY overlay
    # had no control at all.
    window = _dosy_window(qapp, monkeypatch)
    window.canvas_tabs.setCurrentWidget(window.fit_canvas)
    window._load_style_controls_from(window.fit_canvas)
    # A single DOSY map is one plot, not several series.
    assert window.slice_colour_group.isVisible() is False

    first = window.session.spectra[0]
    second = dataclasses.replace(
        first, metadata=dataclasses.replace(first.metadata, raw=dict(first.metadata.raw)))
    _load_spectrum_directly(window, second)
    _select_rows(window, [0, 1])
    window._on_overlay_dosy_clicked()
    window._load_style_controls_from(window.fit_canvas)

    assert window.slice_colour_group.isVisible() or not window.isVisible()
    assert len(window._slice_colour_buttons) == 2


def test_main_window_series_colour_map_is_app_wide(qapp):
    # Unlike the rest of the style panel, which is per plot: this one
    # supplies defaults to several views, and two plots of the same
    # samples should agree without being set up twice.
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.output_colormap_combo.setCurrentText("plasma")

    for canvas in window._all_canvases():
        assert canvas._series_colormap == "plasma"


def test_main_window_overlay_dosy_follows_the_display_panel(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    first = window.session.spectra[0]
    second = dataclasses.replace(
        first, metadata=dataclasses.replace(first.metadata, raw=dict(first.metadata.raw)))
    _load_spectrum_directly(window, second)
    _select_rows(window, [0, 1])
    window._on_overlay_dosy_clicked()

    canvas = window.fit_canvas
    window.dosy_levels_spin.setValue(15)
    assert canvas._dosy_clevels == 15
    # Still an overlay after the redraw, not collapsed back to one map.
    assert len(canvas._dosy_overlay) == 2
    assert len(canvas.ax.collections) == 2


def test_dosy_display_panel_sets_levels_without_re_inverting(qapp, monkeypatch):
    # Reported directly: the ILT had "no way to update the contour
    # levels etc. except for scrolling".
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    assert window.dosy_display_box.isVisible() or not window.isVisible()

    computed = canvas._dosy_result
    window.dosy_levels_spin.setValue(20)
    window.dosy_floor_spin.setValue(10.0)

    assert canvas._dosy_clevels == 20
    assert canvas._dosy_factor == pytest.approx(0.10)
    # Display-only: the map itself is the one already computed, not a
    # fresh (and slow) inversion.
    assert canvas._dosy_result is computed


def test_dosy_floor_box_follows_the_scroll_gesture(qapp, monkeypatch):
    # The panel control and the scroll gesture reach the same quantity,
    # so they must not drift apart.
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    ax = canvas.ax
    bbox = ax.get_window_extent()
    event = types.SimpleNamespace(
        name="scroll_event", canvas=canvas.canvas, inaxes=ax,
        x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 + 0.5 * bbox.height,
        step=1, button="up")
    event.xdata, event.ydata = ax.transData.inverted().transform((event.x, event.y))
    canvas._on_scroll(event)

    assert window.dosy_floor_spin.value() == pytest.approx(canvas._dosy_factor * 100.0)


def test_dosy_d_axis_can_be_pinned_to_the_requested_range(qapp, monkeypatch):
    # Reported directly: the ILT "resets the D axis limits based on the
    # data and ignores the values set by the user".
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    grid = canvas._dosy_result["D"]

    # Seeded from the grid the map was actually computed on -- i.e. the
    # range asked for in the settings dialog.
    assert window.dosy_d_min_spin.value() == pytest.approx(np.log10(grid.min()), abs=1e-6)
    assert window.dosy_d_max_spin.value() == pytest.approx(np.log10(grid.max()), abs=1e-6)

    fitted = canvas.ax.get_ylim()
    window.dosy_fit_d_axis_check.setChecked(False)
    pinned = canvas.ax.get_ylim()

    assert pinned != fitted
    assert pinned[0] == pytest.approx(grid.min(), rel=1e-6)
    assert pinned[1] == pytest.approx(grid.max(), rel=1e-6)

    # And a range typed in by hand is used exactly as given.
    window.dosy_d_min_spin.setValue(-11.0)
    window.dosy_d_max_spin.setValue(-9.0)
    low, high = canvas.ax.get_ylim()
    assert low == pytest.approx(1e-11, rel=1e-6)
    assert high == pytest.approx(1e-9, rel=1e-6)

    # Ticking it back on returns to fitting the axis to the data.
    window.dosy_fit_d_axis_check.setChecked(True)
    assert canvas.ax.get_ylim() == pytest.approx(fitted, rel=1e-6)


def test_dosy_text_is_not_shrunk_when_it_already_fits(qapp, monkeypatch):
    # Reported directly: "ILT axes have tiny text". The DOSY rebuilds
    # its Axes from a fresh gridspec on every redraw, so its axis
    # labels had no valid previous position -- and the layout pass
    # measured that stale position, called it clipped, and shrank every
    # label and tick to the 5 pt floor against a requested 16.
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    canvas.figure.set_size_inches(8, 6)
    canvas._draw_dosy(preserve_view=True)

    assert canvas.ax.xaxis.label.get_fontsize() == pytest.approx(canvas._AXIS_LABELSIZE)
    assert canvas.ax.yaxis.label.get_fontsize() == pytest.approx(canvas._AXIS_LABELSIZE)
    assert all(t.get_fontsize() == pytest.approx(canvas._AXIS_LABELSIZE)
               for t in canvas.ax.get_yticklabels())


def test_dosy_axes_scroll_independently(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    bbox = canvas.ax.get_window_extent()

    x_before, y_before = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    # x strip: below the map
    canvas._on_scroll(types.SimpleNamespace(
        name="scroll_event", canvas=canvas.canvas, inaxes=None,
        x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 - 6, step=1, button="up"))
    assert canvas.ax.get_xlim() != x_before
    assert canvas.ax.get_ylim() == y_before      # y untouched

    x_after = canvas.ax.get_xlim()
    # y strip: to the right of the map, as a contour's own is
    canvas._on_scroll(types.SimpleNamespace(
        name="scroll_event", canvas=canvas.canvas, inaxes=None,
        x=bbox.x1 + 6, y=bbox.y0 + 0.5 * bbox.height, step=1, button="up"))
    assert canvas.ax.get_ylim() != y_before
    assert canvas.ax.get_xlim() == x_after       # x untouched


def test_dosy_y_zoom_stays_positive_on_the_log_axis(qapp, monkeypatch):
    # A log axis zoomed with linear arithmetic walks off toward zero
    # (and matplotlib then refuses to draw it).
    window = _dosy_window(qapp, monkeypatch)
    canvas = window.fit_canvas
    bbox = canvas.ax.get_window_extent()
    for _ in range(6):
        canvas._on_scroll(types.SimpleNamespace(
            name="scroll_event", canvas=canvas.canvas, inaxes=None,
            x=bbox.x1 + 6, y=bbox.y0 + 0.5 * bbox.height, step=1, button="down"))
    low, high = canvas.ax.get_ylim()
    assert low > 0 and high > low


def test_dosy_extract_spectrum_at_a_diffusion_coefficient(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    result = window._dosy_result
    # The D where the fast component actually sits.
    column = int(np.argmin(np.abs(result["ppm"] - 3.0)))
    peak_d = float(result["D"][int(np.argmax(result["Z"][:, column]))])

    before = len(window.session)
    window._on_dosy_extract_requested("spectrum", peak_d)

    assert len(window.session) == before + 1
    extracted = window.session.spectra[before]
    assert extracted.ndim == 1
    # A row of the map is a spectrum: its x axis is chemical shift.
    np.testing.assert_allclose(extracted.axes[0].values("ppm"), result["ppm"])
    assert extracted.metadata.raw["dosy_extract"] == "spectrum"
    # And it carries the fast component's peak, not the slow one's.
    strongest = float(result["ppm"][int(np.argmax(extracted.data))])
    assert strongest == pytest.approx(3.0, abs=0.3)


def test_dosy_extract_profile_at_a_shift(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    result = window._dosy_result

    before = len(window.session)
    window._on_dosy_extract_requested("profile", 7.0)

    extracted = window.session.spectra[before]
    assert extracted.data.size == result["D"].size
    # An index axis, not a frequency one: the values along it are
    # diffusion coefficients, and calling them ppm would be a lie every
    # downstream ppm calculation would then act on.
    assert extracted.axes[0].kind == "index"
    assert extracted.metadata.raw["dosy_extract"] == "profile"
    # The profile peaks at the slow component's own D.
    peak_d = float(result["D"][int(np.argmax(extracted.data))])
    assert peak_d == pytest.approx(1.2e-10, rel=0.4)


def test_dosy_extraction_says_so_when_there_is_nothing_there(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    seen = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: seen.setdefault("title", args[1]))
    before = len(window.session)
    window._on_dosy_extract_requested("profile", 1.0)   # baseline, never inverted
    assert seen.get("title") == "Nothing there"
    assert len(window.session) == before


def test_dosy_state_is_cleared_when_another_view_replaces_it(qapp, monkeypatch):
    # "Is this a DOSY?" has to stay a reliable question for the scroll
    # and right-click handlers.
    window = _dosy_window(qapp, monkeypatch)
    assert window.fit_canvas._dosy_result is not None
    window.fit_canvas.show_spectrum(_make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    assert window.fit_canvas._dosy_result is None


# ============================================================
# Opening the same experiment twice
# ============================================================

def _bruker_1d_folder(tmp_path, name="1"):
    """A minimal loadable Bruker 1D experiment, copied from the real
    test data so every reader path behaves as it does in use."""
    source = os.path.join("test_data", "Bruker Example Data")
    # test_data is gitignored, so a fresh clone has none: skip, like
    # every other real-data test, rather than fail. (From upstream,
    # Leo 1e63c6a -- lost when the merge took this file whole.)
    if not os.path.isdir(source):
        pytest.skip(f"needs {source}")
    matches = MainWindow._scan_for_spectra(source)
    assert matches, "expected a loadable Bruker experiment in test_data"
    target = tmp_path / name
    shutil.copytree(matches[0][0], target)
    return str(target), matches[0][1]


def test_main_window_opening_the_same_folder_twice_switches_instead_of_reloading(qapp, tmp_path):
    path, vendor = _bruker_1d_folder(tmp_path)
    window = MainWindow()

    window._open_paths([path])
    assert len(window.session) == 1

    window._open_paths([path])
    assert len(window.session) == 1
    assert window.spectrum_list.count() == 1
    assert window.spectrum_list.currentRow() == 0
    assert "already open" in window.statusBar().currentMessage()


def test_main_window_reopening_selects_the_existing_row(qapp, tmp_path):
    first, vendor = _bruker_1d_folder(tmp_path, name="1")
    second, _vendor = _bruker_1d_folder(tmp_path, name="2")
    window = MainWindow()

    window._open_paths([first, second])
    assert len(window.session) == 2
    assert window.spectrum_list.currentRow() == 1

    window._open_paths([first])
    assert len(window.session) == 2
    # Switched back to the one already open, rather than appending a
    # third row -- the whole point, since the user's peak table and
    # processing live on that row.
    assert window.spectrum_list.currentRow() == 0


def test_main_window_reopen_does_not_discard_work_on_the_open_copy(qapp, tmp_path):
    path, _vendor = _bruker_1d_folder(tmp_path)
    window = MainWindow()
    window._open_paths([path])
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    assert window.peak_table.rowCount() == 1

    window._open_paths([path])

    assert len(window.session) == 1
    assert window.peak_table.rowCount() == 1


def test_main_window_dropping_an_already_open_folder_does_not_duplicate(qapp, tmp_path):
    path, _vendor = _bruker_1d_folder(tmp_path)
    window = MainWindow()
    window._handle_dropped_paths([path])
    window._handle_dropped_paths([path])
    assert len(window.session) == 1


def test_main_window_browser_activation_does_not_duplicate(qapp, tmp_path):
    path, vendor = _bruker_1d_folder(tmp_path)
    window = MainWindow()
    window._on_browser_spectrum_activated(path, vendor)
    window._on_browser_spectrum_activated(path, vendor)
    assert len(window.session) == 1


def test_main_window_a_removed_spectrum_can_be_opened_again(qapp, tmp_path):
    path, _vendor = _bruker_1d_folder(tmp_path)
    window = MainWindow()
    window._open_paths([path])
    window.spectrum_list.setCurrentRow(0)
    window._on_remove_clicked()
    assert len(window.session) == 0

    window._open_paths([path])
    assert len(window.session) == 1


def test_session_index_for_path_normalises_spellings(tmp_path):
    session = Session()
    path, _vendor = _bruker_1d_folder(tmp_path)
    session.add(path)
    # A trailing slash (what a folder picker returns) and a "." segment
    # are the same experiment.
    assert session.index_for_path(path + "/") == 0
    assert session.index_for_path(os.path.join(path, ".")) == 0
    assert session.index_for_path(str(tmp_path / "nowhere")) is None


def test_session_index_for_path_separates_vendors(tmp_path):
    session = Session()
    path, vendor = _bruker_1d_folder(tmp_path)
    session.add(path, vendor=vendor)
    assert session.index_for_path(path, vendor=vendor) == 0
    # The same folder read by a different reader is a different
    # spectrum, not a duplicate.
    assert session.index_for_path(path, vendor="jcampdx") is None


def test_session_index_for_path_ignores_derived_spectra(qapp, tmp_path):
    session = Session()
    path, _vendor = _bruker_1d_folder(tmp_path)
    spectrum = session.add(path)
    session.add_derived(spectrum, "Measured")
    assert session.index_for_path(path) == 0
    assert session.index_for_path("Measured") is None


def test_main_window_does_not_queue_a_second_read_of_a_loading_folder(qapp, tmp_path, monkeypatch):
    # A background read only reaches the session when it finishes on
    # the GUI thread, so a check on the session alone would let a
    # second drop of the same folder queue a second read of it.
    path, vendor = _bruker_1d_folder(tmp_path)
    window = MainWindow()

    dispatched = []

    class _FakeLoader:
        def load(self, path, vendor, kwargs):
            dispatched.append(path)

    monkeypatch.setattr(window, "_background_loader", lambda: _FakeLoader())
    monkeypatch.setattr(main_window_module, "background_load_enabled", lambda: True)

    window._open_one(path, vendor)
    window._open_one(path, vendor)

    assert dispatched == [path]
    assert "already loading" in window.statusBar().currentMessage()

    # Once the read lands, the path is no longer pending -- and the
    # session check takes over from there.
    window._on_background_load_finished(path, vendor, opennmr_io.load(path, vendor=vendor))
    assert len(window.session) == 1
    window._open_one(path, vendor)
    assert dispatched == [path]
    assert len(window.session) == 1


def test_main_window_list_file_combo_offers_what_is_beside_the_data(qapp, tmp_path):
    # The panel offers the experiment's own list files, labelled by
    # folder so two experiments' identically-named vdlists are
    # distinguishable, with Browse… for anything else.
    sample = tmp_path / "sample"
    expt = sample / "11"
    expt.mkdir(parents=True)
    (expt / "vdlist").write_text("10m\n20m\n40m\n")
    (sample / "extra_delays.txt").write_text("1\n2\n3\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt)))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)  # List file x factor

    labels = [window.list_file_combo.itemText(i) for i in range(window.list_file_combo.count())]
    assert labels[0].startswith("vdlist")
    assert "(11)" in labels[0]
    assert any(text.startswith("extra_delays.txt") for text in labels)
    assert labels[-1] == "Browse…"
    # The likeliest one is selected, so the source works without a click.
    assert window.selected_list_file().endswith("vdlist")


def test_main_window_values_preview_shows_what_will_be_fit(qapp):
    delays = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(delays))
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    window.manual_delays_edit.setText("10, 20, 40, 80, 160, 320")  # ms

    text = window.values_preview_label.text()
    assert text.startswith("6 values [ms]:")
    # Displayed in the unit they were typed in, not silently converted.
    assert text.endswith("10, 20, 40, 80, 160, 320")


def test_main_window_values_preview_elides_only_a_real_gap(qapp):
    # With exactly six values, "…, 320" used to sit between two
    # consecutive entries and read as though a gap had been elided.
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.01] * 6))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText("1, 2, 3, 4, 5, 6")
    assert "…" not in window.values_preview_label.text()

    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.01] * 7))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText("1, 2, 3, 4, 5, 6, 7")
    text = window.values_preview_label.text()
    assert text.endswith("1, 2, 3, 4, 5, …, 7")


def test_main_window_values_preview_flags_a_length_mismatch(qapp):
    # A list that does not match the slice count used to surface only
    # as an IndexError from _get_delays, after Fit Curve was clicked.
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04]))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText("1, 2")
    # Too few: flagged, and the way out is named.
    text = window.values_preview_label.text()
    assert "⚠ 3 slices" in text and "set Slices to 2" in text


def test_main_window_values_preview_does_not_warn_about_a_longer_list(qapp):
    # A list longer than the data is Bruker's own convention, not a
    # problem: the surplus is simply unused.
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04]))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText("1, 2, 3, 4, 5")
    text = window.values_preview_label.text()
    assert "⚠" not in text
    assert "first 3 used" in text


def test_main_window_values_preview_reports_a_source_error_in_place(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04]))
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText("")
    assert window.values_preview_label.text().startswith("Values: Enter a list of values")


def test_main_window_list_factor_can_be_an_experiment_parameter(qapp, tmp_path):
    # The user's own case: a CPMG echo-count list times TAU, so a T2
    # series is fitted without doing the arithmetic by hand.
    expt = tmp_path / "cpmg"
    expt.mkdir()
    (expt / "echo_valueslist.txt").write_text("2\n4\n8\n16\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04, 0.08])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, vendor="xpulse", source_path=str(expt),
        raw={"exp_params": {"TAU": 1500.0, "P90": 8.78}}))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)  # List file x factor
    # TAU is recorded in microseconds, and the Unit says the whole
    # product is in microseconds -- so the factor needs no conversion
    # at all, which is the point of the unit applying to the product.
    _set_delay_unit(window, "µs")
    window.list_factor_edit.setText("TAU*2")

    values = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(values, [0.006, 0.012, 0.024, 0.048])

    # The panel says what it resolved to and which parameter it used.
    assert "Factor = 3000" in window.list_factor_value_label.text()
    assert "TAU=1500" in window.list_factor_value_label.text()

    # The same series, asked for in seconds: the factor then carries
    # the conversion instead.
    _set_delay_unit(window, "s")
    window.list_factor_edit.setText("TAU*2*1e-6")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False),
        [0.006, 0.012, 0.024, 0.048])


def test_main_window_list_factor_reports_an_unknown_parameter(qapp, tmp_path):
    expt = tmp_path / "cpmg"
    expt.mkdir()
    (expt / "echo_valueslist.txt").write_text("2\n4\n8\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, vendor="xpulse", source_path=str(expt),
        raw={"exp_params": {"TAU": 1500.0}}))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)
    window.list_factor_edit.setText("TAO*2")

    # Named, not silently treated as a free parameter -- there is
    # nothing to fit a factor against.
    assert "TAO" in window.list_factor_value_label.text()
    assert "TAU" in window.list_factor_value_label.text()  # what is available
    with pytest.raises(ValueError, match="does not record"):
        window._resolve_independent_values(spectrum, is_diffusion=False)


def test_main_window_list_factor_empty_means_one(qapp, tmp_path):
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n20m\n40m\n")
    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt)))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)
    assert window.list_factor_edit.text() == ""
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.01, 0.02, 0.04])


def test_main_window_list_file_does_not_carry_over_between_experiments(qapp, tmp_path):
    # "the list file should be from the relevant folder, not remembered
    # from the previous" -- selecting a second experiment must offer
    # *its* list, not re-offer the first one's.
    first = tmp_path / "11"
    first.mkdir()
    (first / "vdlist").write_text("10m\n20m\n40m\n")
    second = tmp_path / "12"
    second.mkdir()
    (second / "vdlist").write_text("100m\n200m\n400m\n")

    window = MainWindow()
    for path in (first, second):
        spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
        spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
            spectrum.metadata, source_path=str(path)))
        _load_spectrum_directly(window, spectrum)

    window.delay_source_combo.setCurrentIndex(3)  # List file x factor
    assert window.selected_list_file() == str(second / "vdlist")
    paths = [window.list_file_combo.itemData(i) for i in range(window.list_file_combo.count())]
    assert str(first / "vdlist") not in paths

    window.spectrum_list.setCurrentRow(0)
    window.delay_source_combo.setCurrentIndex(3)
    assert window.selected_list_file() == str(first / "vdlist")
    paths = [window.list_file_combo.itemData(i) for i in range(window.list_file_combo.count())]
    assert str(second / "vdlist") not in paths


def test_main_window_list_file_choice_survives_reselecting_the_same_spectrum(qapp, tmp_path):
    # The other half of the same rule: a deliberate pick on *this*
    # experiment must not be thrown away by a refresh of its own combo.
    expt = tmp_path / "11"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n20m\n40m\n")
    (expt / "other_list.txt").write_text("1\n2\n3\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt)))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)

    chosen = window.list_file_combo.findData(str(expt / "other_list.txt"))
    assert chosen >= 0
    window.list_file_combo.setCurrentIndex(chosen)
    window._refresh_list_file_candidates()
    assert window.selected_list_file() == str(expt / "other_list.txt")


def test_main_window_exports_the_curve_fit_as_values_vs_x(qapp, tmp_path, monkeypatch):
    # "I need the curve fit results to be exportable as integral/peak
    # height vs whatever x is."
    true_t2 = 0.05
    delays = np.geomspace(0.001, 0.3, 8)
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(
        delays, x0=5.0, fwhm=0.05, T2=true_t2))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)  # Manual list
    _set_delay_unit(window, "ms")
    window.manual_delays_edit.setText(", ".join(f"{d * 1e3:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # Exponential decay (T2)

    assert not window.export_curve_fit_button.isEnabled()
    window._on_fit_spectrum_clicked()
    window._on_fit_curve_button_clicked()
    assert window.export_curve_fit_button.isEnabled()

    target = tmp_path / "curve.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args, **kwargs: (str(target), "CSV files (*.csv)"))
    window._on_export_curve_fit_clicked()

    lines = target.read_text().splitlines()
    # **The parameters come first**, on request: "Export curve fit needs
    # to include the fit report parameters, like D/T1/T2 and the fit
    # quality". The series follows its own section header.
    assert lines[0] == "Curve fit parameters"
    param_header = lines[1].split(",")
    assert "T2 (s)" in param_header and "R^2" in param_header
    fitted_t2 = float(lines[2].split(",")[param_header.index("T2 (s)")])
    assert fitted_t2 == pytest.approx(true_t2, rel=0.05)

    series_at = lines.index("Curve fit series")
    rows = list(csv.reader(lines[series_at + 1:]))
    header, body = rows[0], rows[1:]
    # The x column names the unit the panel is showing, and the measured
    # column names which Value source produced it.
    assert header[0] == "delay (ms)"
    assert any("integral" in name for name in header[1:])
    assert any("fit" in name for name in header[1:])
    assert len(body) == len(delays)
    # Written in that unit, not in seconds. (rtol covers the %.6g the
    # test itself typed the list with, not the export.)
    np.testing.assert_allclose([float(row[0]) for row in body], delays * 1e3, rtol=1e-5)
    # The measured column decays the way the series does.
    measured = [float(row[1]) for row in body]
    assert measured[0] > measured[-1]


def test_main_window_export_curve_fit_with_nothing_fitted_says_so(qapp, monkeypatch):
    window = MainWindow()
    shown = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: shown.update(text=args[2]))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args, **kwargs: pytest.fail("should not ask for a filename"))
    window._on_export_curve_fit_clicked()
    assert "Fit Curve" in shown["text"]


# ============================================================
# The delay unit reaches the whole curve-fitting panel
# ============================================================
# Report: "while the unit is selected and does the calculation
# correctly, the curve fit and resultant values just take the values
# naively in seconds". The conversion always reached the fit; what
# stayed in seconds was everything the user then *looked at and typed
# into* -- the Fit tab's x-axis and the guess table.

def _fitted_t2_window(qapp, unit="ms"):
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(
        delays, x0=5.0, fwhm=0.05, T2=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, unit)
    scale = {"s": 1.0, "ms": 1e3, "µs": 1e6}[unit]
    window.manual_delays_edit.setText(", ".join(f"{d * scale:.8g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # Exponential decay (T2)
    window._on_fit_spectrum_clicked()
    window._on_fit_curve_button_clicked()
    return window, delays


def test_main_window_fit_plot_is_drawn_in_the_selected_unit(qapp):
    window, delays = _fitted_t2_window(qapp, "ms")
    assert window.fit_canvas.ax.get_xlabel() == "Delay (ms)"
    # The x-axis reads in the same numbers that were typed, not 0.001.
    np.testing.assert_allclose(
        window.fit_canvas.ax.lines[-1].get_xdata(), delays * 1e3, rtol=1e-6)
    # The stored fit is untouched -- the unit is a display transform.
    np.testing.assert_allclose(
        window._relaxation_fit_results[0]["t"], delays, rtol=1e-6)
    assert window._relaxation_fit_results[0]["params"]["T2"] == pytest.approx(0.05, rel=0.1)


def test_main_window_changing_the_unit_redraws_the_fit_without_refitting(qapp):
    window, delays = _fitted_t2_window(qapp, "ms")
    before = window._relaxation_fit_results[0]["params"]["T2"]

    _set_delay_unit(window, "s")
    assert window.fit_canvas.ax.get_xlabel() == "Delay (s)"
    np.testing.assert_allclose(
        window.fit_canvas.ax.lines[-1].get_xdata(), delays, rtol=1e-6)
    # Same fit, redrawn -- not refitted on rescaled data.
    assert window._relaxation_fit_results[0]["params"]["T2"] == before


def test_main_window_diffusion_fit_plot_is_never_rescaled(qapp):
    # A b-value has no prefix convention; the Unit combo must not touch it.
    window, b_values, _spectrum = _pfg_series_window(qapp, true_D=2.0e-9)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)  # Stejskal-Tanner
    window._on_fit_spectrum_clicked()
    window._on_fit_curve_button_clicked()

    assert "b-value" in window.fit_canvas.ax.get_xlabel()
    np.testing.assert_allclose(
        window.fit_canvas.ax.lines[-1].get_xdata(), b_values, rtol=1e-6)


def test_main_window_guess_table_times_are_in_the_selected_unit(qapp):
    window, _delays = _fitted_t2_window(qapp, "ms")
    window.guess_group.setChecked(True)
    window._refresh_guess_table(force=True)

    rows = {window.guess_table.item(r, 0).text(): r for r in range(window.guess_table.rowCount())}
    # The automatic T2 guess, shown in ms rather than as 0.018.
    assert float(window.guess_table.item(rows["T2"], 1).text()) > 1.0
    assert "times in ms" in window.guess_group.title()

    # A typed guess is read back in that unit.
    window.guess_table.setItem(rows["T2"], 1, QTableWidgetItem("40"))
    window._store_current_guess_edits()
    p0, _bounds, _fixed = window._read_guess_table(["M0", "T2"])
    assert p0[1] == pytest.approx(0.04)


def test_main_window_guess_table_leaves_non_time_parameters_alone(qapp):
    # An amplitude is on the data's own intensity scale and a stretch
    # exponent is dimensionless -- scaling either by a time unit would
    # be nonsense.
    window, _delays = _fitted_t2_window(qapp, "ms")
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("stretched_decay"))
    window.guess_group.setChecked(True)
    window._refresh_guess_table(force=True)

    rows = {window.guess_table.item(r, 0).text(): r for r in range(window.guess_table.rowCount())}
    window.guess_table.setItem(rows["M0"], 1, QTableWidgetItem("3"))
    window.guess_table.setItem(rows["beta"], 1, QTableWidgetItem("0.8"))
    window.guess_table.setItem(rows["T2"], 1, QTableWidgetItem("40"))
    window._store_current_guess_edits()

    p0, _bounds, _fixed = window._read_guess_table(["M0", "T2", "beta"])
    assert p0[0] == pytest.approx(3.0)      # amplitude, unscaled
    assert p0[1] == pytest.approx(0.04)     # T2, ms -> s
    assert p0[2] == pytest.approx(0.8)      # beta, dimensionless


def test_main_window_adopt_fit_as_guess_writes_in_the_selected_unit(qapp):
    window, _delays = _fitted_t2_window(qapp, "ms")
    window._on_adopt_fit_as_guess_clicked()
    rows = {window.guess_table.item(r, 0).text(): r for r in range(window.guess_table.rowCount())}
    # ~50 ms, not 0.05.
    assert float(window.guess_table.item(rows["T2"], 1).text()) == pytest.approx(50.0, rel=0.1)


def test_main_window_guess_bound_error_quotes_the_displayed_unit(qapp):
    # An error naming numbers the user cannot find on screen is worse
    # than no error.
    window, _delays = _fitted_t2_window(qapp, "ms")
    window.guess_group.setChecked(True)
    window._refresh_guess_table(force=True)
    rows = {window.guess_table.item(r, 0).text(): r for r in range(window.guess_table.rowCount())}
    window.guess_table.setItem(rows["T2"], 1, QTableWidgetItem("900"))
    window.guess_table.setItem(rows["T2"], 2, QTableWidgetItem("1"))
    window.guess_table.setItem(rows["T2"], 3, QTableWidgetItem("100"))
    window._store_current_guess_edits()

    with pytest.raises(ValueError, match=r"900 ms"):
        window._read_guess_table(["M0", "T2"])


def test_main_window_excluding_a_point_uses_the_displayed_axis(qapp):
    # The click arrives in display units; matching it against the
    # stored seconds series would always pick the same end point.
    window, delays = _fitted_t2_window(qapp, "ms")
    target = delays[4]
    result = window._relaxation_fit_results[0]
    y_at_target = float(result["y"][int(np.argmin(np.abs(result["t"] - target)))])

    window._on_fit_point_exclude_requested(target * 1e3, y_at_target)

    excluded = window._excluded_relaxation_indices.get(0, set())
    assert len(excluded) == 1
    original = window._relaxation_fit_original[0]
    assert original["t"][next(iter(excluded))] == pytest.approx(target, rel=1e-6)


def test_main_window_a_relaxation_fit_takes_the_canvas_back_from_enmr(qapp):
    # Regression, and it hung the suite rather than failing it. The
    # right-click owner has to be *set* by every writer of this canvas,
    # not only read by one of them: with only the eNMR side setting it,
    # a relaxation exclusion after any eNMR fit went down the eNMR
    # path, refitted a spectrum that was not an eNMR series, and put up
    # a modal error box with nothing waiting to dismiss it.
    window, delays = _fitted_t2_window(qapp, "ms")
    assert window._fit_canvas_owner == "relaxation"

    window._fit_canvas_owner = "enmr"          # as if an eNMR fit ran
    window._display_relaxation_peak(0)
    assert window._fit_canvas_owner == "relaxation"

    target = delays[4]
    result = window._relaxation_fit_results[0]
    y_at_target = float(result["y"][int(np.argmin(np.abs(result["t"] - target)))])
    window._on_fit_point_exclude_requested(target * 1e3, y_at_target)

    assert len(window._excluded_relaxation_indices.get(0, set())) == 1
    assert window._enmr_excluded == set()


def test_main_window_enmr_refit_after_exclusion_never_blocks_on_a_dialog(qapp,
                                                                          monkeypatch):
    # A refit the user did not ask for reports through the status bar.
    window = MainWindow()
    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *args, **kwargs: shown.append(args))
    window._enmr_quiet = True
    window._enmr_problem("something went wrong")
    assert shown == []
    window._enmr_quiet = False
    window._enmr_problem("something went wrong")
    assert len(shown) == 1


def test_main_window_curve_fit_export_x_matches_the_plot(qapp, tmp_path, monkeypatch):
    window, delays = _fitted_t2_window(qapp, "ms")
    target = tmp_path / "curve.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args, **kwargs: (str(target), "CSV files (*.csv)"))
    window._on_export_curve_fit_clicked()

    # The file now opens with the fitted parameters; the series
    # follows its own section header -- see curve_fit_to_csv.
    lines = target.read_text().splitlines()
    series_at = lines.index("Curve fit series")
    rows = list(csv.reader(lines[series_at + 1:]))
    assert rows[0][0] == "delay (ms)"
    np.testing.assert_allclose(
        [float(r[0]) for r in rows[1:]],
        window.fit_canvas.ax.lines[-1].get_xdata(), rtol=1e-6)


def test_main_window_list_file_bare_values_are_in_the_selected_unit(qapp, tmp_path):
    # "when I set ms it seems to multiply the value by 1e3 because it
    # assumes that the read value is s, whereas the unit should set the
    # unit for the read value". A file holding 10, 20, 40 meant
    # milliseconds and was read as 10, 20, 40 seconds.
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "delays_valueslist.txt").write_text("10\n20\n40\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt)))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)

    _set_delay_unit(window, "ms")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.01, 0.02, 0.04])
    # And it reads back as what is in the file, not 1000x it.
    assert window.values_preview_label.text().endswith("10, 20, 40")

    _set_delay_unit(window, "µs")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [1e-5, 2e-5, 4e-5])

    _set_delay_unit(window, "s")
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [10.0, 20.0, 40.0])


def test_main_window_list_file_with_its_own_units_ignores_the_combo(qapp, tmp_path):
    # A vdlist writing "10m" has already said what it means; applying
    # the combo on top would scale an absolute time a second time.
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n20m\n40m\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt)))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(3)

    for unit in ("s", "ms", "µs"):
        _set_delay_unit(window, unit)
        np.testing.assert_allclose(
            window._resolve_independent_values(spectrum, is_diffusion=False),
            [0.01, 0.02, 0.04], err_msg=f"rescaled a suffixed list under {unit}")


def test_main_window_auto_source_is_never_rescaled_by_the_combo(qapp, tmp_path):
    # Auto is the vendor's own values, in the vendor's own unit -- it is
    # not something the user is declaring a unit for.
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "vdlist").write_text("0.01\n0.02\n0.04\n")

    spectrum = _make_pseudo2d_relaxation_spectrum([0.01, 0.02, 0.04])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, vendor="bruker", source_path=str(expt)))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.delay_source_combo.setCurrentIndex(0)  # Auto

    for unit in ("s", "ms", "µs"):
        _set_delay_unit(window, unit)
        values = window._resolve_independent_values(spectrum, is_diffusion=False)
        values = values["seconds"] if isinstance(values, dict) else values
        np.testing.assert_allclose(values, [0.01, 0.02, 0.04],
                                    err_msg=f"rescaled Auto under {unit}")


def test_main_window_multi_peak_summary_quotes_amplitudes_as_shares(qapp):
    # The compact per-peak table exists to compare peaks against each
    # other, and an intensity on the data's own arbitrary scale is the
    # one number in it that cannot be compared.
    delays = np.geomspace(0.002, 0.5, 14)
    ppm = np.linspace(0.0, 10.0, 1200)
    from opennmr.processing.fitting import lorentzian
    from opennmr.processing.relaxation import biexponential_decay

    data = np.zeros((len(delays), ppm.size))
    for i, t in enumerate(delays):
        data[i] = (lorentzian(ppm, 3.0, 0.05, biexponential_decay(t, 7.0, 0.02, 3.0, 0.3))
                   + lorentzian(ppm, 7.0, 0.05, biexponential_decay(t, 2.0, 0.02, 8.0, 0.3)))
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(delays) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=3.0, fwhm=0.05, eta=0.0)
    window._add_peak_row(shift=7.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.8g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("biexponential_decay"))

    window._on_fit_spectrum_clicked()
    window._on_fit_curve_button_clicked()

    summary = window.curve_fit_summary_label.text()
    assert "A1=" in summary and "%" in summary
    # The two peaks have opposite dominant components, which is exactly
    # the comparison the shares make readable.
    peak_lines = [line for line in summary.splitlines() if line.startswith("Peak")]
    assert len(peak_lines) == 2
    shares = [float(line.split("A1=")[1].split("%")[0]) for line in peak_lines]
    assert shares[0] > 50.0 > shares[1]


# ============================================================
# Limiting how many slices the analysis uses
# ============================================================

def test_main_window_slice_limit_trims_what_the_fit_uses(qapp):
    # "There should be an easy way to limit the number of slices used."
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(
        delays, x0=5.0, fwhm=0.05, T2=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.8g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)

    window.slice_limit_spin.setValue(6)
    window._on_fit_spectrum_clicked()
    window._on_fit_curve_button_clicked()

    # Six points fitted, not ten -- and they are the first six.
    result = window._relaxation_fit_results[0]
    assert len(result["t"]) == 6
    np.testing.assert_allclose(result["t"], delays[:6], rtol=1e-6)


def test_main_window_slice_limit_of_zero_means_every_slice(qapp):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.01] * 8)
    _load_spectrum_directly(window, spectrum)
    assert window.slice_limit_spin.value() == 0
    assert window.slice_limit_spin.specialValueText() == "all"
    assert window._fit_slice_count(spectrum) == 8
    assert window._fit_slice_rows(spectrum) == list(range(1, 9))


def test_main_window_slice_limit_larger_than_the_series_is_not_an_error(qapp):
    # Usually means the box was set for a different experiment.
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.01] * 5)
    _load_spectrum_directly(window, spectrum)
    window.slice_limit_spin.setValue(50)
    assert window._fit_slice_count(spectrum) == 5


def test_main_window_slice_limit_is_siloed_per_spectrum(qapp):
    window = MainWindow()
    for _ in range(2):
        _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.01] * 8))
    window.slice_limit_spin.setValue(4)

    window.spectrum_list.setCurrentRow(0)
    assert window.slice_limit_spin.value() == 0   # this one was never limited
    window.spectrum_list.setCurrentRow(1)
    assert window.slice_limit_spin.value() == 4


def test_main_window_curve_fit_survives_a_longer_value_list(qapp):
    # The whole point: a list longer than the data must just work.
    delays = np.geomspace(0.001, 0.3, 8)
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(
        delays, x0=5.0, fwhm=0.05, T2=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    # Twelve values for eight slices.
    longer = list(delays) + [0.4, 0.5, 0.6, 0.7]
    window.manual_delays_edit.setText(", ".join(f"{d:.8g}" for d in longer))
    window.relaxation_model_combo.setCurrentIndex(2)

    window._on_fit_spectrum_clicked()
    window._on_fit_curve_button_clicked()

    result = window._relaxation_fit_results[0]
    assert len(result["t"]) == 8
    assert result["params"]["T2"] == pytest.approx(0.05, rel=0.1)


def test_main_window_dosy_uses_only_the_limited_slices(qapp, monkeypatch):
    window, b_values, _spectrum = _pfg_series_window(qapp, true_D=2.0e-9)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.slice_limit_spin.setValue(6)

    captured = {}
    real = processing_ilt.dosy_map

    def spy(b, data, ppm, **kwargs):
        captured["n_b"] = len(b)
        captured["rows"] = data.shape[0]
        return real(b, data, ppm, **kwargs)

    monkeypatch.setattr(processing_ilt, "dosy_map", spy)
    monkeypatch.setattr(_DosySettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_dosy_clicked()

    # The b-values and the data matrix are cut together -- a map built
    # from more rows than there are b-values would be meaningless.
    assert captured["n_b"] == 6
    assert captured["rows"] == 6


# ============================================================
# eNMR sub-tab
# ============================================================

ENMR_CALIBRATION = os.path.join(
    "test_data", "eNMR", "Diff50_1H_eNMR_Calibration_2024-05-13", "21")


def _enmr_window(qapp, path=ENMR_CALIBRATION):
    window = MainWindow()
    window._load(path, vendor="bruker")
    return window


def test_main_window_has_an_enmr_subtab(qapp):
    window = MainWindow()
    side = _side_tab_names(window)
    applications_tab = window.side_tabs.widget(side.index("Applications"))
    titles = [s.title() for s in applications_tab.findChildren(CollapsibleSection)]
    assert "eNMR" in titles


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_reads_voltages_and_constants_from_the_experiment(qapp):
    window = _enmr_window(qapp)
    # The vdlist conversion is a setting, so what it resolves to is shown.
    assert "voltages [V]" in window.enmr_voltage_label.text()
    assert "0, 0, 8, 8, 16, 16" in window.enmr_voltage_label.text()
    # And the PFG constants come from diff.xml, with provenance.
    constants = window.enmr_constants_label.text()
    assert "0.3 T/m" in constants and "δ = 1 ms" in constants and "Δ = 100 ms" in constants
    assert "diff.xml" in constants


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_fit_recovers_the_literature_mobility(qapp):
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    window.enmr_gap_spin.setValue(38.5)

    window._on_fit_enmr_clicked()

    # The literature value for TMA+, through the GUI's own load path
    # (which autophases, unlike the processing-level test) -- so a few
    # percent rather than an exact string.
    assert window._enmr_result["mobility"] == pytest.approx(3.79e-8, rel=0.05)
    # Negative: the vdlist puts the positive polarity first, so the
    # slope is read the literal way round. The panel's cell-polarity
    # setting -- which way the + electrode faces, recorded nowhere --
    # is what turns it into a positive mobility for a cation.
    assert window._enmr_result["slope"] == pytest.approx(-7.90e-3, rel=0.05)
    assert window.enmr_polarity_combo.currentData() == -1
    assert window._enmr_result["cell_polarity"] == -1
    report = window.enmr_results_label.text()
    assert "m²/Vs" in report and "electrode gap = 38.5 mm" in report
    # Plotted against volts, and never rescaled by the delay-unit combo.
    assert window.fit_canvas.ax.get_xlabel() == "Voltage (V)"


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_without_a_gap_reports_the_slope_only(qapp):
    # The gap is the one number no file records; without it the slope
    # is the result, which is what the data determines.
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    assert window.enmr_gap_spin.value() == 0
    assert "slope only" in window.enmr_gap_spin.specialValueText()

    window._on_fit_enmr_clicked()

    report = window.enmr_results_label.text()
    assert "rad/V" in report
    assert "Mobility not computed" in report


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_combines_gradient_reversed_repeats(qapp):
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)

    assert not window.enmr_keep_repeat_button.isEnabled()
    window._on_fit_enmr_clicked()
    assert window.enmr_keep_repeat_button.isEnabled()

    window._on_enmr_keep_repeat_clicked()
    assert not window.enmr_combine_button.isEnabled()   # one is not a pair
    window._on_enmr_keep_repeat_clicked()
    assert window.enmr_combine_button.isEnabled()

    window._on_enmr_combine_clicked()
    report = window.enmr_results_label.text()
    assert "spread across 2 repeats" in report

    window._on_enmr_clear_repeats_clicked()
    assert not window.enmr_combine_button.isEnabled()


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_unfolds_the_pairs_onto_a_signed_axis(qapp):
    # The recorded axis folds every point onto its partner
    # (0, 0, 8, 8, ...). Unfolded, the series is one ramp through the
    # origin, so the zero crossing is on screen.
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    assert window.enmr_unfold_checkbox.isChecked()

    window._on_fit_enmr_clicked()

    plotted = sorted(window.fit_canvas.ax.lines[1].get_xdata())
    assert plotted[0] == pytest.approx(-56.0)
    assert plotted[-1] == pytest.approx(56.0)
    assert len(plotted) == 16                      # every slice, not 8 pairs
    assert window.fit_canvas.ax.get_xlabel() == "Voltage (V)"
    # A phase shift in degrees, and no longer labelled "Intensity".
    assert window.fit_canvas.ax.get_ylabel() == "Phase shift (°)"
    # Quoted against the field-off measurement: the 0 V pair averages
    # to zero. The two members differ by their own noise -- they are
    # not forced onto it, which would hide that difference.
    branches = window._enmr_result["branches"]
    at_zero = branches["voltages"] == 0
    assert branches["phases"][at_zero].mean() == pytest.approx(0.0, abs=1e-12)
    assert np.degrees(np.abs(branches["phases"][at_zero])).max() < 2.0
    assert branches["referenced_to"] == "V = 0"


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_unfold_toggles_without_refitting(qapp):
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    window._on_fit_enmr_clicked()
    slope = window._enmr_result["slope"]

    window.enmr_unfold_checkbox.setChecked(False)

    # Folded: the pair-differenced measurement against |V|.
    folded = sorted(window.fit_canvas.ax.lines[1].get_xdata())
    assert len(folded) == 8 and folded[0] == pytest.approx(0.0)
    assert window.fit_canvas.ax.get_xlabel() == "|Voltage| (V)"
    # A display choice only -- the fit itself is untouched.
    assert window._enmr_result["slope"] == slope


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_reports_the_two_polarity_arms(qapp):
    # Whether averaging the polarities was justified is the reason to
    # run both, so it is in the report without a second click.
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    window._on_fit_enmr_clicked()

    report = window.enmr_results_label.text()
    assert "polarity arms" in report
    assert "did not reverse" in report
    assert "Zero crossing" in report
    # The calibration standard's arms agree, so no warning on it.
    assert "DISAGREE" not in report


def test_main_window_enmr_unfold_toggle_is_harmless_before_any_fit(qapp):
    window = MainWindow()
    window.enmr_unfold_checkbox.setChecked(False)
    window.enmr_unfold_checkbox.setChecked(True)
    assert window._enmr_result is None


LIAC_PAIR = os.path.join("test_data", "Diff50_eNMR_Li_Ac_6p25M_2024-09-03")


@pytest.mark.skipif(not os.path.isdir(LIAC_PAIR), reason="needs the real eNMR dataset")
def test_main_window_enmr_averages_a_real_gradient_reversed_pair(qapp):
    # The second pairing: two whole experiments, one at +g and one at
    # -g. 560/561 is such a pair, and it disagrees.
    window = MainWindow()
    for expt in ("560", "561"):
        window._load(os.path.join(LIAC_PAIR, expt), vendor="bruker")
    window.enmr_low_spin.setValue(1.34)
    window.enmr_high_spin.setValue(1.60)

    for row in (0, 1):
        window.spectrum_list.setCurrentRow(row)
        window._on_fit_enmr_clicked()
        window._on_enmr_keep_repeat_clicked()

    # Each repeat carries the gradient sign its own diff.xml recorded.
    gradients = [entry["gradient"] for entry in window._enmr_repeats]
    assert np.sign(gradients[0]) == -np.sign(gradients[1])

    window._on_enmr_combine_clicked()
    report = window.enmr_results_label.text()
    assert "g (T/m)" in report and "average" in report
    assert "DISAGREE" in report

    # All three on one axes: both directions behind their average.
    labels = [line.get_label() for line in window.fit_canvas.ax.lines]
    assert sum("g = " in str(label) for label in labels) == 2
    assert any("Average of 2 gradient directions" in str(label) for label in labels)
    assert window.fit_canvas.ax.get_ylabel() == "Phase shift, sign-corrected (°)"


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_falls_back_to_slopes_when_ramps_cannot_be_averaged(qapp):
    # Repeats measured at different voltages cannot be averaged point
    # by point; combining their slopes still works, and says so.
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    window._on_fit_enmr_clicked()
    window._on_enmr_keep_repeat_clicked()
    window._on_enmr_keep_repeat_clicked()
    # Shorten one of them so the voltage axes no longer match.
    window._enmr_repeats[1] = {
        **window._enmr_repeats[1],
        "voltages": window._enmr_repeats[1]["voltages"][:8],
        "phases": window._enmr_repeats[1]["phases"][:8],
    }

    window._on_enmr_combine_clicked()

    report = window.enmr_results_label.text()
    assert "Slopes only" in report
    assert "spread across 2 repeats" in report


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_voltage_cutoff_trims_the_fit_and_keeps_the_points(qapp):
    # The ramp goes nonlinear at the top of the range, so the highest
    # points need keeping off the fit -- but on the plot, where the
    # deviation stays visible.
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    assert window.enmr_max_voltage_spin.value() == 0
    assert "all points" in window.enmr_max_voltage_spin.specialValueText()

    window.enmr_max_voltage_spin.setValue(32.0)
    window._on_fit_enmr_clicked()

    branches = window._enmr_result["branches"]
    assert int(branches["used"].sum()) == 10       # 0-32 V at both polarities
    assert np.all(np.abs(branches["voltages"][branches["used"]]) <= 32)
    report = window.enmr_results_label.text()
    assert "|V| ≤ 32 V" in report

    # Fitted points as circles, the trimmed ones as grey crosses.
    labels = [line.get_label() for line in window.fit_canvas.ax.lines]
    assert "Excluded" in labels
    excluded = window.fit_canvas.ax.lines[labels.index("Excluded")]
    assert len(excluded.get_xdata()) == 6          # 40, 48, 56 V, both polarities


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_cell_polarity_flips_only_the_mobility(qapp):
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    window.enmr_gap_spin.setValue(38.5)
    window._on_fit_enmr_clicked()
    measured = window._enmr_result

    window.enmr_polarity_combo.setCurrentIndex(
        window.enmr_polarity_combo.findData(1))
    window._on_fit_enmr_clicked()

    assert window._enmr_result["slope"] == pytest.approx(measured["slope"])
    assert window._enmr_result["mobility"] == pytest.approx(-measured["mobility"])
    assert "cell polarity = +1" in window.enmr_results_label.text()


@pytest.mark.skipif(not os.path.isdir(ENMR_CALIBRATION), reason="needs the real eNMR dataset")
def test_main_window_enmr_right_click_excludes_one_point_not_its_pair(qapp):
    # A single polarised electrode distorts one polarity only, so the
    # damaged points come one at a time.
    window = _enmr_window(qapp)
    window.enmr_low_spin.setValue(3.18)
    window.enmr_high_spin.setValue(3.42)
    window._on_fit_enmr_clicked()
    branches = window._enmr_result["branches"]
    assert branches["balanced"] is True

    # Right-click near the +56 V point.
    target = int(np.argmax(branches["voltages"]))
    window._on_fit_point_exclude_requested(
        float(branches["voltages"][target]), float(branches["phases"][target]))

    assert window._enmr_excluded == {target}
    after = window._enmr_result["branches"]
    assert after["used"][target] is np.False_ or not after["used"][target]
    # Its −56 V partner is still in the fit.
    partner = int(np.argmin(after["voltages"]))
    assert after["used"][partner]
    assert after["balanced"] is False
    # The half-difference lost that pair, and the report says to read
    # the arms instead.
    assert window._enmr_result["broken_pairs"] == 1
    assert "could not" in window.enmr_results_label.text()

    window._on_fit_points_reset_requested()
    assert window._enmr_excluded == set()
    assert window._enmr_result["branches"]["balanced"] is True


def test_main_window_enmr_right_click_does_not_reach_into_a_stale_relaxation_fit(qapp):
    # _fit_mode is on for every fit plot, so without an owner the eNMR
    # plot's right-click would silently re-fit whatever relaxation run
    # happened earlier in the session.
    window = MainWindow()
    assert window._fit_canvas_owner is None
    window._on_fit_point_exclude_requested(1.0, 1.0)
    assert window._excluded_relaxation_indices == {}


def test_main_window_enmr_needs_a_pseudo_2d(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    shown = {}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: shown.update(text=args[2]))
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    window._on_fit_enmr_clicked()
    assert "pseudo-2D" in shown["text"]


# ============================================================
# DOSY rows beside the map; no baseline overlay on a derived trace
# ============================================================

def _populated_dosy_d(window):
    """A diffusion coefficient the inversion actually put signal at --
    most of the D grid is empty by construction."""
    result = window.fit_canvas._dosy_result
    z = np.asarray(result["Z"], dtype=float)
    grid = np.asarray(result["D"], dtype=float)
    populated = [i for i in range(z.shape[0]) if np.any(z[i, :])]
    return float(grid[populated[len(populated) // 2]])


def test_dosy_row_can_be_drawn_beside_the_map(qapp, monkeypatch):
    """Explicit user request: "when I extract a line from the DOSY, I
    would also like the option to plot alongside with the dashed line
    like in normal 2D contours"."""
    from opennmr.plotting.twod import SLICE_PANEL_GID

    window = _dosy_window(qapp, monkeypatch)
    d_value = _populated_dosy_d(window)
    before = len(window.session.spectra)

    window._on_dosy_extract_requested("spectrum_panel", d_value)
    window.fit_canvas.canvas.draw()

    # Drawn beside the map, not added to the spectrum list -- that is
    # what distinguishes it from the plain "Extract Spectrum" action.
    assert len(window.session.spectra) == before
    assert len(window.fit_canvas._dosy_slices) == 1
    panel = [a for a in window.fit_canvas.figure.axes if a.get_gid() == SLICE_PANEL_GID]
    assert len(panel) == 1
    # And the canvas knows it as a slice panel, so the scroll/cursor
    # handlers treat it the way a 2D contour's own panel is treated.
    assert window.fit_canvas._proj_axes["slices"] is panel[0]


def test_dosy_rows_are_siloed_on_the_spectrum_and_restored_with_the_map(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window._on_dosy_extract_requested("spectrum_panel", _populated_dosy_d(window))
    assert len(spectrum.metadata.raw["dosy_slices"]) == 1

    window.fit_canvas.clear()
    assert window.fit_canvas._dosy_slices == []

    window._restore_dosy_for(spectrum)
    window.fit_canvas.canvas.draw()
    assert len(window.fit_canvas._dosy_slices) == 1


def test_the_same_dosy_row_twice_is_not_drawn_twice(qapp, monkeypatch):
    """A repeat is a mis-click, not a request for two identical dashed
    lines on top of each other."""
    window = _dosy_window(qapp, monkeypatch)
    d_value = _populated_dosy_d(window)
    window._on_dosy_extract_requested("spectrum_panel", d_value)
    window._on_dosy_extract_requested("spectrum_panel", d_value)
    assert len(window.fit_canvas._dosy_slices) == 1


def test_plain_dosy_extraction_still_adds_a_spectrum(qapp, monkeypatch):
    """The panel mode is an addition, not a replacement."""
    window = _dosy_window(qapp, monkeypatch)
    before = len(window.session.spectra)
    window._on_dosy_extract_requested("spectrum", _populated_dosy_d(window))
    assert len(window.session.spectra) == before + 1
    assert window.fit_canvas._dosy_slices == []


def test_no_baseline_overlay_on_an_extracted_trace(qapp, monkeypatch):
    """Reported directly: "when the 1D is extracted, I shouldn't see the
    baseline dashed line on the plot." A derived trace is computed from
    already-processed data -- there is no correction in the pipeline for
    the overlay to be previewing."""
    window = _dosy_window(qapp, monkeypatch)
    window.baseline_preview_checkbox.setChecked(True)

    window._on_dosy_extract_requested("spectrum", _populated_dosy_d(window))
    derived = len(window.session.spectra) - 1
    assert window.session.spectra[derived].metadata.vendor == "derived"

    window.spectrum_list.setCurrentRow(derived)
    window._refresh_baseline_preview()
    window.canvas.canvas.draw()

    overlay = [ln for ln in window.canvas.ax.lines
               if ln.get_gid() in (window.canvas._BASELINE_PREVIEW_GID,
                                   window.canvas._BASELINE_GHOST_GID)]
    assert overlay == []


# ============================================================
# Excluded slices: one set, every fitting method
# ============================================================

def test_excluded_slices_are_siloed_on_the_spectrum(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    assert MainWindow.excluded_slices(spectrum) == set()

    window.set_excluded_slices(spectrum, {2, 5})
    assert MainWindow.excluded_slices(spectrum) == {2, 5}
    assert spectrum.metadata.raw["excluded_slices"] == [2, 5]


def test_included_rows_drops_the_excluded_ones(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    n = window._fit_slice_count(spectrum)
    window.set_excluded_slices(spectrum, {1, 3})
    rows = window._included_rows(spectrum, n)
    assert 1 not in rows and 3 not in rows
    assert len(rows) == n - 2
    assert rows == sorted(rows)          # order preserved, not just a set


def test_the_slice_list_shows_one_row_per_slice_and_ticks_the_included(qapp, monkeypatch):
    """A checkable list, not a text field of indices: the question is
    "which of these is bad", which you answer by looking at them."""
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window._refresh_excluded_slice_list(spectrum)

    assert window.excluded_slice_group.isVisible() or not window.isVisible()
    assert window.excluded_slice_list.count() == window._fit_slice_count(spectrum)
    assert all(window.excluded_slice_list.item(i).checkState() == Qt.CheckState.Checked
               for i in range(window.excluded_slice_list.count()))

    window.set_excluded_slices(spectrum, {0})
    assert window.excluded_slice_list.item(0).checkState() == Qt.CheckState.Unchecked
    assert window.excluded_slice_list.item(1).checkState() == Qt.CheckState.Checked


def test_unticking_a_slice_writes_it_onto_the_spectrum(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window.spectrum_list.setCurrentRow(0)
    window._refresh_excluded_slice_list(spectrum)

    window.excluded_slice_list.item(2).setCheckState(Qt.CheckState.Unchecked)
    assert MainWindow.excluded_slices(spectrum) == {2}

    window._on_include_all_slices_clicked()
    assert MainWindow.excluded_slices(spectrum) == set()


def test_the_slice_list_is_hidden_for_anything_without_slices(qapp, tmp_path):
    """A group of dead controls beside a 1D spectrum reads as broken
    rather than as not-yet-applicable."""
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window._refresh_excluded_slice_list(window.session.spectra[0])
    assert not window.excluded_slice_group.isVisible()


def test_dosy_inverts_only_the_included_slices(qapp, monkeypatch):
    """An ILT fits the whole matrix, so a spoiled gradient step has no
    per-peak exclusion to hide behind -- this is the only way to drop
    one. Explicit user request that it reach SCORE and the ILT."""
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    n = window._fit_slice_count(spectrum)

    seen = {}
    real_dosy_map = processing_ilt.dosy_map

    def capture(b_values, data, ppm, **kwargs):
        seen["rows"] = len(b_values)
        seen["data_rows"] = data.shape[0]
        return real_dosy_map(b_values, data, ppm, **kwargs)

    monkeypatch.setattr(processing_ilt, "dosy_map", capture)
    window.set_excluded_slices(spectrum, {0, 1})
    window._on_dosy_clicked()

    # Both the b-values and the data matrix lost the same two rows --
    # dropping one without the other is the misalignment this is
    # written to avoid.
    assert seen["rows"] == n - 2
    assert seen["data_rows"] == n - 2


def test_dosy_refuses_when_almost_everything_is_excluded(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    n = window._fit_slice_count(spectrum)
    window.set_excluded_slices(spectrum, set(range(n - 1)))

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    called = []
    monkeypatch.setattr(processing_ilt, "dosy_map", lambda *a, **k: called.append(1))
    window._on_dosy_clicked()

    assert warned and not called


def test_curve_fit_filtering_drops_series_entries_but_keeps_the_value_list_whole(qapp, monkeypatch):
    """**The values are looked up by slice number, not by position.**
    ``NMR2D_fit_relaxation`` reads ``vdlist[fit["index"] - 1]``, and a
    kept slice keeps its original index -- so re-packing the values to
    positions 0..n would leave the last slice pointing past the end.
    Reported as "a mismatch between the number of integrals and number
    of points to fit"."""
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window.set_excluded_slices(spectrum, {2, 6})

    results = [{"index": i + 1} for i in range(10)]
    values = np.arange(10, dtype=float) * 0.5
    kept_results, kept_values = window._drop_excluded_from_series(
        spectrum, results, values, 10)

    # The series loses the excluded slices...
    assert [r["index"] for r in kept_results] == [1, 2, 4, 5, 6, 8, 9, 10]
    # ...and the value list stays whole, so slice 10 still finds its
    # own value. A longer list than the fit needs is explicitly fine.
    assert list(kept_values) == list(values)


def test_curve_fit_with_excluded_slices_actually_fits(qapp, monkeypatch):
    """The end-to-end shape of the bug: excluding a slice and refitting
    raised rather than fitting."""
    t = np.geomspace(0.01, 0.5, 10)
    fits = [{"index": i + 1, "integrals": [float(10.0 * np.exp(-ti / 0.1))]}
            for i, ti in enumerate(t)]
    keep = [i for i in range(10) if i not in (0, 2)]

    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window.set_excluded_slices(spectrum, {0, 2})
    kept_results, kept_values = window._drop_excluded_from_series(
        spectrum, fits, t, 10)

    # No ValueError about slices without a value.
    results = processing_relaxation.NMR2D_fit_relaxation(
        kept_results, model="exponential_decay", vdlist=kept_values,
        show_report=False)
    assert len(results[0]["t"]) == len(keep)
    assert results[0]["params"]["T2"] == pytest.approx(0.1, rel=0.01)


def test_curve_fit_filtering_leaves_a_values_dict_untouched(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    spectrum = window.session.spectra[0]
    window.set_excluded_slices(spectrum, {1})
    results = [{"index": i + 1} for i in range(4)]
    values = {"seconds": np.array([0.0, 1.0, 2.0, 3.0]), "unit": "s"}

    kept_results, kept_values = window._drop_excluded_from_series(
        spectrum, results, values, 4)
    assert [r["index"] for r in kept_results] == [1, 3, 4]
    assert kept_values is values


def test_curve_fit_series_filtering_is_a_no_op_for_a_compare_stack(qapp, monkeypatch):
    """A Compare-tab stack has no single owning spectrum, so positions
    in it mean nothing to any one spectrum's exclusion set."""
    window = _dosy_window(qapp, monkeypatch)
    results = [{"slice": i} for i in range(4)]
    values = np.arange(4, dtype=float)
    kept_results, kept_values = window._drop_excluded_from_series(None, results, values, 4)
    assert kept_results is results and kept_values is values


def test_score_resolves_only_the_included_slices(qapp, monkeypatch):
    """SCORE fits the whole data matrix, so a spoiled slice pulls on
    every resolved component at once -- and a per-peak right-click
    exclusion has nothing to act on for it."""
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    spectrum = window.session.spectra[0]
    n = window._fit_slice_count(spectrum)

    seen = {}
    real_score = processing_score.score

    def capture(x_values, data, **kwargs):
        seen["x"] = len(x_values)
        seen["rows"] = data.shape[0]
        return real_score(x_values, data, **kwargs)

    monkeypatch.setattr(processing_score, "score", capture)
    window.set_excluded_slices(spectrum, {0, 2})
    window._on_score_clicked()

    assert seen["x"] == n - 2
    assert seen["rows"] == n - 2


def test_score_refuses_when_almost_everything_is_excluded(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    spectrum = window.session.spectra[0]
    n = window._fit_slice_count(spectrum)
    window.set_excluded_slices(spectrum, set(range(n - 1)))

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    called = []
    monkeypatch.setattr(processing_score, "score", lambda *a, **k: called.append(1))
    window._on_score_clicked()

    assert warned and not called


def test_no_baseline_overlay_on_the_spectrum_tab_for_plain_1d_data(qapp, tmp_path, monkeypatch):
    """The checkbox defaults to on, and the Spectrum canvas used to be
    in the preview list -- so every ordinary 1D got a dashed blue curve
    over it by default. Reported twice: "in 1D data, I still see the
    baseline dashed blue line on the spectrum tab".

    The earlier derived-spectrum gate did not cover this: an ordinary
    loaded spectrum is not derived."""
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")
    assert window.baseline_preview_checkbox.isChecked()      # still on by default

    overlay = [ln for ln in window.canvas.ax.lines
               if ln.get_gid() in (window.canvas._BASELINE_PREVIEW_GID,
                                   window.canvas._BASELINE_GHOST_GID)]
    assert overlay == []


def test_the_baseline_overlay_still_reaches_the_processing_preview(qapp, tmp_path, monkeypatch):
    """Moved, not removed -- the speculative view is where an
    uncommitted correction belongs."""
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=512), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    preview = window.processing_preview_canvas

    assert [ln for ln in preview._pivot_marker_ax.lines
            if ln.get_gid() == preview._BASELINE_PREVIEW_GID]


def test_gradient_axis_from_the_file_drives_the_probe_calibration(qapp):
    """The per-axis probe store exists because the coils differ; it is
    only any use if the axis the experiment actually used is known."""
    path = "test_data/xpulse_PFG/4"
    if not os.path.isdir(path):
        pytest.skip("test_data/xpulse_PFG not present")
    window = MainWindow()
    window._load(path, vendor="xpulse")
    spectrum = window.session.spectra[window._selected_index()]

    assert spectrum.metadata.raw["gradient_axis"] == "y"
    assert window._gradient_axis(spectrum) == "y"
    # And the combo the user sees agrees with the file.
    assert window.pfg_gradient_axis_combo.currentData() == "y"
    assert window.imaging_gradient_axis_combo.currentData() == "y"


def _calibration_window(qapp, known_d=2.299e-9, true_g_max=55.0, n=16, corrupt=None):
    """A PFG series on a standard of known D, spectrally fitted and
    ready to calibrate. ``corrupt`` : row index to spoil, so a test can
    check that excluding it recovers the right answer."""
    relative = np.linspace(0.02, 0.95, n)
    b = processing_diffusion.b_values_from_relative(
        relative, true_g_max, 2e-3, 100e-3, nucleus="1H", shape="smsq10",
        max_gradient_unit="G/cm")
    from opennmr.processing.fitting import gaussian
    ppm = np.linspace(0.0, 10.0, 700)
    amplitudes = np.exp(-known_d * b)
    if corrupt is not None:
        amplitudes[corrupt] *= 3.0          # a spoiled scan
    data = np.array([gaussian(ppm, 5.0, 0.05, a) for a in amplitudes])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True,
                              raw={"gradient_relative": relative}),
    )
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._on_fit_spectrum_clicked()
    return window, true_g_max


def test_gradient_calibration_honours_the_excluded_slices(qapp, monkeypatch):
    """Asked directly: "Is the calibrate gradient function also able to
    use the selection of slices used in the curve fit?" It is a fit
    like any other, and a spoiled gradient step damages it the same way
    -- arguably more, since the whole probe's G_max comes off it."""
    window, true_g_max = _calibration_window(qapp, corrupt=5)
    monkeypatch.setattr(_GradientCalibrationDialog, "exec",
                        lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    spectrum = window.session.spectra[0]

    # With the bad point in, the answer is off.
    window._on_calibrate_gradient_clicked()
    spoiled = window.pfg_max_gradient_spin.value()
    assert abs(spoiled - true_g_max) / true_g_max > 0.02

    # Excluding it recovers the true maximum.
    window.set_excluded_slices(spectrum, {5})
    window._on_calibrate_gradient_clicked()
    assert window.pfg_max_gradient_spin.value() == pytest.approx(true_g_max, rel=0.02)


def test_gradient_calibration_says_how_many_slices_it_used(qapp, monkeypatch):
    """A calibration quietly computed from fewer points than the series
    holds is worth seeing next to the number it produced."""
    window, _g = _calibration_window(qapp)
    monkeypatch.setattr(_GradientCalibrationDialog, "exec",
                        lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    window._on_calibrate_gradient_clicked()
    assert "excluded" not in window.gradient_calibration_label.text()

    window.set_excluded_slices(window.session.spectra[0], {2, 9})
    window._on_calibrate_gradient_clicked()
    text = window.gradient_calibration_label.text()
    assert "From 14 of 16 slices" in text and "2 excluded" in text


def test_gradient_calibration_refuses_when_almost_everything_is_excluded(qapp, monkeypatch):
    window, _g = _calibration_window(qapp)
    window.set_excluded_slices(window.session.spectra[0], set(range(15)))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    opened = []
    monkeypatch.setattr(_GradientCalibrationDialog, "exec",
                        lambda self: opened.append(1) or QDialog.DialogCode.Accepted)

    window._on_calibrate_gradient_clicked()
    # Refused before the settings dialog, not after filling it in.
    assert warned and not opened


def test_the_score_report_lands_with_score_not_in_peak_picking(qapp, monkeypatch):
    """``analysis_results_label`` is built by
    ``_build_peak_picking_subtab`` and is that sub-tab's own results
    display, so a resolution used to print its report on a different
    tab -- beside a peak table it explicitly does not use. Reported
    directly: "why is the SCORE report in the peak picking tab? it
    should be with the SCORE in analysis"."""
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    before = window.analysis_results_label.text()

    window._on_score_clicked()

    assert "SCORE" in window.score_results_label.text()
    # Peak Picking's own label is left exactly as it was.
    assert window.analysis_results_label.text() == before


def test_the_score_report_is_siloed_and_hidden_with_its_panel(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    spectrum = window.session.spectra[0]
    report = window.score_results_label.text()
    assert report

    # Siloed with the spectrum, like every other per-spectrum panel --
    # it left analysis_results_label, so it needs its own slot or a
    # reselection loses it.
    state = window._capture_analysis_state()
    assert state["score_results_text"] == report
    spectrum.metadata.raw["analysis_state"] = state

    window.score_results_label.setText("")
    window._restore_analysis_state(spectrum)
    assert window.score_results_label.text() == report


# ============================================================
# Results land with the analysis that produced them
# ============================================================

def test_every_analysis_report_has_its_own_home(qapp):
    """``analysis_results_label`` is built by
    ``_build_peak_picking_subtab`` and is that sub-tab's own display,
    but four analyses used to write to it -- so a DOSY, a SCORE, a
    curve fit and a gradient calibration all printed their results on
    the Peak Picking tab. Each now has its own label."""
    window = MainWindow()
    for name in ("score_results_label", "dosy_results_label",
                 "curve_fit_summary_label", "gradient_calibration_label"):
        assert hasattr(window, name), name


def test_the_dosy_summary_lands_with_dosy(qapp, monkeypatch):
    window = _dosy_window(qapp, monkeypatch)
    before = window.analysis_results_label.text()
    assert window.dosy_results_label.text()
    assert window.analysis_results_label.text() == before


def test_the_calibration_result_lands_with_the_gradient_controls(qapp, monkeypatch):
    window, _g = _calibration_window(qapp)
    monkeypatch.setattr(_GradientCalibrationDialog, "exec",
                        lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    before = window.analysis_results_label.text()

    window._on_calibrate_gradient_clicked()
    assert "Maximum gradient" in window.gradient_calibration_label.text()
    assert window.analysis_results_label.text() == before


def test_all_four_reports_survive_a_reselection(qapp, monkeypatch):
    """They left analysis_results_label, which was the only one the
    analysis state captured -- each needs its own slot or a reselection
    loses it."""
    window = _dosy_window(qapp, monkeypatch)
    window.score_results_label.setText("score text")
    window.curve_fit_summary_label.setText("curve text")
    window.gradient_calibration_label.setText("calibration text")

    state = window._capture_analysis_state()
    for key, expected in (("score_results_text", "score text"),
                          ("curve_fit_summary_text", "curve text"),
                          ("gradient_calibration_text", "calibration text")):
        assert state[key] == expected
    assert state["dosy_results_text"] == window.dosy_results_label.text()


# ============================================================
# The status bar says what is open, not just what just happened
# ============================================================

def test_status_bar_states_the_selected_experiment(qapp, monkeypatch):
    """It was transient-only -- 32 showMessage calls and nothing else --
    so between operations the space said nothing. Explicit user
    request: "make better use of the status bar at the bottom beyond
    just loading"."""
    window = _dosy_window(qapp, monkeypatch)
    text = window._status_experiment.text()
    assert "1H" in text
    assert "pseudo-2D" in text
    assert window._status_session.text() == "1 spectrum"


def test_status_bar_is_empty_with_nothing_open(qapp):
    window = MainWindow()
    assert window._status_experiment.text() == ""
    assert window._status_session.text() == ""


def test_status_bar_context_is_permanent_not_transient(qapp, monkeypatch):
    """Permanent widgets, so a load/export/fit message takes the
    left-hand space and this stays put beside it rather than being
    wiped by it."""
    window = _dosy_window(qapp, monkeypatch)
    before = window._status_experiment.text()
    window.statusBar().showMessage("Loading something…")
    assert window._status_experiment.text() == before
    window.statusBar().clearMessage()
    assert window._status_experiment.text() == before


def test_f1_processing_preview_shows_an_interferogram_and_a_spectrum(qapp, tmp_path):
    """"F1 processing doesn't show fid and spectrum, can this be done?"

    It can: the readers already stash every row's F2-processed data as
    ``raw["fid"]`` for a genuine 2D, and a column of that *is* F1's
    time-domain series -- the one the F1 transform runs across, and
    the one Truncate (F1) and Zero-fill (F1) are decisions about."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=32), vendor="xpulse")
    spectrum = window.session.spectra[window._selected_index()]

    interferogram, dt_f1 = window._f1_interferogram(spectrum)
    assert interferogram is not None
    # One point per F1 increment of the *time* domain, so it matches
    # the row count of the stashed F2-processed rows -- not the
    # (TPPI-halved) row count of the transformed spectrum.
    assert len(interferogram) == np.asarray(spectrum.metadata.raw["fid"]).shape[0]
    assert dt_f1 == pytest.approx(spectrum.metadata.raw["fid_dt_f1"])
    # Real-valued: an amplitude-modulated F1 carries its t1 information
    # in the real part, the imaginary part being F2's quadrature.
    assert np.allclose(np.imag(interferogram), 0.0)


def test_f1_interferogram_is_absent_rather_than_wrong_without_cached_rows(qapp):
    """A 1D, or a 2D loaded from processed data with no F1 time domain
    cached, has no interferogram -- the panel is skipped, not faked."""
    window = MainWindow()
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    bare = NMRSpectrum(
        data=np.zeros((4, 8)), axes=[NMRAxis(kind="index", index=np.arange(4)),
                                      NMRAxis(kind="frequency", ppm=np.arange(8.0))],
        metadata=NMRMetadata(vendor="xpulse", raw={}),
    )
    assert window._f1_interferogram(bare) == (None, None)


def test_f1_window_controls_drive_the_live_rebuild(qapp, tmp_path):
    """"I don't think the windowing is being applied to the indirect
    dimension." It wasn't -- F1 had no window at all for X-Pulse, and
    for Bruker only its own recorded `proc2s` WDW, with nothing the
    user could choose. Its own controls, not F2's re-targeted through
    dim_combo: F1 truncate and zero-fill are already separate widgets,
    and a heavy sine-bell on 200 increments beside a light exponential
    on 2048 points is the normal case."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=16, n2=32), vendor="xpulse")
    index = window._selected_index()
    spectrum = window.session.spectra[index]

    assert window.window_f1_combo.isEnabled()
    assert window._resolve_f1_window() is None          # "None" by default
    before = np.array(spectrum.data, copy=True)

    window.window_f1_combo.setCurrentIndex(window.window_f1_combo.findData("sine"))
    window.ssb_f1_spin.setValue(2.0)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)

    assert window._resolve_f1_window() == ("sine", {"ssb": 2.0})
    assert not np.allclose(before, window.session.spectra[index].data)


def test_f1_window_choice_survives_reselection(qapp, tmp_path):
    """Persisted and restored like every other live control, so
    switching experiments doesn't quietly drop it."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=16, n2=32, name="a.jdx"),
                 vendor="xpulse")
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=16, n2=32, name="b.jdx"),
                 vendor="xpulse")

    window.spectrum_list.setCurrentRow(0)
    window.window_f1_combo.setCurrentIndex(window.window_f1_combo.findData("qsine"))
    window.ssb_f1_spin.setValue(2.0)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)

    window.spectrum_list.setCurrentRow(1)
    assert window.window_f1_combo.currentData() == "none"   # B never chose one
    window.spectrum_list.setCurrentRow(0)
    assert window.window_f1_combo.currentData() == "qsine"
    assert window.ssb_f1_spin.value() == pytest.approx(2.0)


def test_the_f1_preview_draws_the_f1_window_envelope(qapp, tmp_path):
    """"Certainly I don't see the windowing in the processing view."
    The envelope is F2's own overlay, applied to F1's panel -- without
    it there is no way to see what Window F1 is doing, or that it is
    doing nothing."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=16, n2=32), vendor="xpulse")

    assert window._compute_f1_window_envelope(1e-3, 16) is None   # nothing chosen
    window.window_f1_combo.setCurrentIndex(window.window_f1_combo.findData("sine"))
    window.ssb_f1_spin.setValue(2.0)

    envelope = window._compute_f1_window_envelope(1e-3, 16)
    assert envelope is not None and len(envelope) == 16
    assert envelope.max() == pytest.approx(1.0)
    # A cosine-bell (ssb=2) starts at its maximum and decays.
    assert envelope[0] > envelope[-1]


def test_saved_f1_window_is_folded_into_the_reader_kwargs():
    """A reopened spectrum must read with the same F1 window it was
    left with, or it comes back looking different until a control is
    touched -- the "two pipelines quietly diverge" failure again."""
    kwargs = MainWindow._saved_processing_read_kwargs(
        {"live_window_f1": "sine", "live_ssb_f1": 2.0})
    assert kwargs["window_f1"] == "sine"
    assert kwargs["window_kwargs_f1"] == {"ssb": 2.0}

    # "none" maps to nothing at all, not to a zero-width exponential
    # the way F2's does: F1's default really is no window.
    assert "window_f1" not in MainWindow._saved_processing_read_kwargs(
        {"live_window_f1": "none"})
    assert "window_f1" not in MainWindow._saved_processing_read_kwargs({})


def test_contour_settings_survive_a_reprocess_of_the_same_spectrum(qapp, tmp_path):
    """"The contour levels and view shouldn't reset when processing."
    A live reprocess writes into the same NMRSpectrum object and
    redraws it; resetting the noise floor, level count, scaling and
    exponent there threw away deliberate choices on every settle --
    while _redraw_2d_preserving_view was carefully preserving the zoom
    right beside it."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=32), vendor="xpulse")
    spectrum = window.session.spectra[window._selected_index()]
    canvas = window.canvas

    canvas.set_contour_levels(23, "linear", 3.5, 0.02)
    canvas.show_spectrum(spectrum)          # the redraw a reprocess performs

    assert canvas._contour_clevels == 23
    assert canvas._contour_level_mode == "linear"
    assert canvas._contour_power == pytest.approx(3.5)
    assert canvas._contour_factor == pytest.approx(0.02)


def test_a_different_spectrum_does_reset_the_contour_settings(qapp, tmp_path):
    """The other half of the rule: settings follow the spectrum they
    were chosen for, so selecting a different one starts clean."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=32, name="a.jdx"),
                 vendor="xpulse")
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=32, name="b.jdx"),
                 vendor="xpulse")
    canvas = window.canvas
    canvas.set_contour_levels(23, "linear", 3.5, 0.02)

    canvas.show_spectrum(window.session.spectra[0])
    assert canvas._contour_clevels == MplCanvas._CONTOUR_CLEVELS_DEFAULT
    assert canvas._contour_level_mode == MplCanvas._CONTOUR_LEVEL_MODE_DEFAULT


def test_plot_style_carries_contour_scaling_and_power(qapp, tmp_path):
    """"The plot style for a 2D contour should include number of
    contours and the power scaling." The count was already there; the
    spacing was only behind the right-click dialog, which split one
    decision across two places -- the count alone decides nothing
    about whether a weak peak beside a strong one is visible."""
    window = MainWindow()
    window._load(_make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=32), vendor="xpulse")

    window.output_contour_mode_combo.setCurrentIndex(
        window.output_contour_mode_combo.findData("linear"))
    assert window.canvas._contour_level_mode == "linear"
    # The exponent is meaningless for linear spacing, so it greys out.
    assert window.output_contour_power_spin.isEnabled() is False

    window.output_contour_mode_combo.setCurrentIndex(
        window.output_contour_mode_combo.findData("power"))
    window.output_contour_power_spin.setValue(2.5)
    assert window.canvas._contour_level_mode == "power"
    assert window.canvas._contour_power == pytest.approx(2.5)

    # And reopening the dialog reads them back off the live canvas.
    window.canvas.set_contour_levels(9, "linear", 4.0, 0.05)
    window._load_style_controls_from(window.canvas)
    assert window.output_contour_levels_spin.value() == 9
    assert window.output_contour_mode_combo.currentData() == "linear"
    assert window.output_contour_power_spin.value() == pytest.approx(4.0)


def test_the_f1_processing_preview_draws_the_pivot_line(qapp, tmp_path, monkeypatch):
    """The single-panel F1 view always drew the Phase 1 pivot; the
    dual-panel one added with the interferogram returned without it,
    so the pivot vanished from F1 the moment that panel appeared."""
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8, f1_nuc="13C"),
                 vendor="bruker")
    window.process_indirect_button.setChecked(True)
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.dim_combo.setCurrentIndex(1)                      # F1 (indirect)

    canvas = window.processing_preview_canvas
    assert canvas._processing_preview_fid_ax is not None     # the dual panel is up
    assert canvas._pivot_line is not None, "no pivot line on the F1 preview"


def _csa_spectrum(d_iso=100.0, aniso=200.0, asym=0.3, fwhm=5.0, lo=-200.0, hi=500.0, n=512):
    """A synthetic CSA powder pattern as an NMRSpectrum, ppm descending."""
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.notebook.simulations import csa_definition, make_static_csa_spectrum

    ascending = np.linspace(lo, hi, n)
    simulated = make_static_csa_spectrum(csa_definition(d_iso, aniso, asym),
                                         ascending, fwhm=fwhm, n_theta=120, n_phi=240)
    ppm = ascending[::-1]
    return NMRSpectrum(
        data=simulated["real_spec"][::-1],
        axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 100.0)],
        metadata=NMRMetadata(vendor="test", nucleus="13C", sfo1=100.0),
    )


def _window_with_csa(spectrum):
    window = MainWindow()
    window.session.add_derived(spectrum, "csa")
    window.spectrum_list.addItem("csa")
    window.spectrum_list.setCurrentRow(0)
    window.csa_orientations_spin.setValue(800)
    window.csa_render_orientations_spin.setValue(2000)
    return window


def test_the_csa_fit_recovers_a_known_powder_lineshape(qapp):
    """The CSA sub-tab, end to end: table -> bridge -> Euan's
    multi-site fitter -> curve on the Fit canvas -> report.

    FWHM is deliberately not asserted tightly: the fit runs on a
    coarse powder grid on purpose (the loop is what every Jacobian
    evaluation costs), so it absorbs the grid's graininess as extra
    broadening. The tensor parameters are what a CSA fit is for."""
    window = _window_with_csa(_csa_spectrum())
    window._add_csa_site(("A", "85", "170", "0.45", "7"))
    window._on_csa_fit_clicked()

    result = window._csa_fit_result
    assert result is not None
    assert result["popt"][0] == pytest.approx(100.0, abs=3.0)   # d_iso
    assert result["popt"][1] == pytest.approx(200.0, abs=8.0)   # Delta
    assert result["popt"][2] == pytest.approx(0.30, abs=0.08)   # eta
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert "CSA FIT REPORT" in window.csa_results_label.text().upper()


def test_a_clipped_powder_pattern_is_warned_about(qapp):
    """**Measured, not hypothetical.** A powder pattern that runs off
    the edge of the fitted range fits back with a badly shrunken span
    -- 126 ppm against a true 200 -- and nothing else says so, because
    the residual *inside* the window stays small. The fitted span is
    the number people take away from a CSA fit."""
    # Same tensor, but a window that cuts the high-frequency shoulder.
    window = _window_with_csa(_csa_spectrum(lo=-100.0, hi=300.0))
    window._add_csa_site(("A", "85", "170", "0.45", "7"))
    window._on_csa_fit_clicked()

    assert window._csa_fit_result is not None
    assert "clipped" in window.csa_results_label.text().lower()


def test_the_csa_site_table_reports_bad_input_by_row(qapp):
    window = _window_with_csa(_csa_spectrum())
    with pytest.raises(ValueError, match="at least one CSA site"):
        window._read_csa_sites()

    window._add_csa_site(("A", "not a number", "170", "0.45", "7"))
    with pytest.raises(ValueError, match="Site row 1"):
        window._read_csa_sites()


def test_the_csa_site_table_defaults_its_label(qapp):
    window = _window_with_csa(_csa_spectrum())
    window._add_csa_site(("", "0", "100", "0", "5"))
    assert window._read_csa_sites()[0]["label"] == "Site 1"


def test_the_mas_rate_is_seeded_from_the_file(qapp, tmp_path, monkeypatch):
    """Bruker records the spinning rate as MASR in acqus and the
    reader already carries it to metadata.raw, so the finite-MAS
    control needs somewhere to read it rather than new plumbing.
    Seeded, never forced -- a rate recorded at setup and not used is
    common enough that the box stays editable."""
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=8), vendor="bruker")
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw.setdefault("acqus", {})["MASR"] = 25000

    window.csa_mas_spin.setValue(0.0)
    window._refresh_csa_controls(spectrum)
    assert window.csa_mas_spin.value() == pytest.approx(25000.0)

    # Already set by hand: left alone.
    window.csa_mas_spin.setValue(12000.0)
    window._refresh_csa_controls(spectrum)
    assert window.csa_mas_spin.value() == pytest.approx(12000.0)


def test_the_finite_mas_lineshape_refuses_a_zero_rate(qapp, monkeypatch):
    """Better than simulating a nonsense sideband pattern at 0 Hz."""
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

    window = _window_with_csa(_csa_spectrum())
    window._add_csa_site(("A", "85", "170", "0.45", "7"))
    window.csa_static_combo.setCurrentIndex(
        window.csa_static_combo.findData(False))
    window.csa_mas_spin.setValue(0.0)
    window._on_csa_fit_clicked()

    assert warned, "no warning for a finite-MAS fit with no spinning rate"
    assert "spinning rate" in warned[-1][2]


def test_the_powder_grid_is_expressed_the_way_each_scheme_wants(qapp):
    """ZCW takes one orientation count; the rectangular grid takes a
    theta/phi pair, split in the 1:2 ratio the simulator's own
    defaults use."""
    assert MainWindow._csa_grid_kwargs("zcw", 5000) == {"n_zcw": 5000}
    rectangular = MainWindow._csa_grid_kwargs("rectangular", 5000)
    assert rectangular["n_phi"] == 2 * rectangular["n_theta"]


# ============================================================
# Simulations tab: forward-simulate a CSA or quadrupolar powder
# pattern, exportable to Load/Save/Compare -- explicit user request,
# and the last (7th) of the launcher buttons. Genuinely new, not moved
# from an old tab, so tested more thoroughly than a pure re-parenting.
# ============================================================

def _fast_csa_kwargs(window) -> None:
    """Keep every simulation test fast: a coarse powder grid is fine
    when the point is checking wiring/shape/export, not lineshape
    fidelity."""
    window.sim_csa_orientations_spin.setValue(200)
    window.sim_quad_orientations_spin.setValue(200)
    window.sim_csa_points_spin.setValue(256)
    window.sim_quad_points_spin.setValue(256)


def test_main_window_simulations_is_its_own_top_level_button(qapp):
    window = MainWindow()
    side = _side_tab_names(window)
    assert "Simulations" in side
    # After Fit, before Dynamics (toolbar order, user request 2026-09-27).
    assert side.index("Fitting") < side.index("Simulations") < side.index("Relaxation")

    simulations_tab = window.side_tabs.widget(side.index("Simulations"))
    sections = simulations_tab.findChildren(CollapsibleSection)
    # Diffusion (PFG) and Shaped pulse (Bloch): ported from upstream;
    # pNMR (CIF): the notebook's CIF shift simulation (test_gui_pnmr.py).
    # J-coupling first (test_gui_jcoupling.py); Czjzek distribution after
    # Quadrupolar (test_gui_czjzek_display.py).
    assert [s.title() for s in sections] == ["J-coupling", "CSA", "Quadrupolar",
                                             "Czjzek distribution",
                                             "Combined Hamiltonian", "Diffusion (PFG)",
                                             "Shaped pulse (Bloch)", "pNMR (CIF)"]

    csa_section = _enclosing_collapsible_section(window.sim_csa_diso_spin)
    quad_section = _enclosing_collapsible_section(window.sim_quad_cq_spin)
    assert csa_section.title() == "CSA"
    assert quad_section.title() == "Quadrupolar"
    assert csa_section is not quad_section


def test_main_window_simulation_export_buttons_start_disabled(qapp):
    window = MainWindow()
    assert not window.sim_csa_export_button.isEnabled()
    assert not window.sim_quad_export_button.isEnabled()


def test_main_window_simulate_csa_static_refuses_an_unrecognised_nucleus(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_csa_nucleus_edit.setText("not-a-nucleus")
    window._on_simulate_csa_clicked()

    assert warned, "no warning for an unrecognised nucleus"
    assert "nucleus" in warned[-1][2].lower()
    assert not window.sim_csa_export_button.isEnabled()


def test_main_window_simulate_csa_static_produces_an_exportable_spectrum(qapp):
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_diso_spin.setValue(50.0)
    window.sim_csa_aniso_spin.setValue(20.0)
    window.sim_csa_skeweta_spin.setValue(0.3)
    # Wide enough axis that the static pattern (centred on d_iso, width
    # set by the small aniso above) is not clipped -- this test is
    # about the export path, not the axis-width warning.
    window.sim_csa_ppm_min_spin.setValue(-50.0)
    window.sim_csa_ppm_max_spin.setValue(150.0)

    window._on_simulate_csa_clicked()

    assert window.sim_csa_export_button.isEnabled()
    assert "static" in window.sim_csa_results_label.text()
    assert "clipped" not in window.sim_csa_results_label.text()

    before = len(window.session.spectra)
    window._on_export_simulated_csa_clicked()
    assert len(window.session.spectra) == before + 1

    exported = window.session.spectra[-1]
    assert exported.ndim == 1
    assert exported.data.shape == (window.sim_csa_points_spin.value(),)
    assert exported.metadata.nucleus == "13C"
    # Session.add_derived always overwrites vendor to "derived" --
    # documented, intentional behaviour so FID-reprocess controls
    # (gated on the real vendor names) stay hidden for it -- not a
    # "simulated" vendor string, whatever the spectrum arrived with.
    assert exported.metadata.vendor == "derived"
    assert exported.axes[0].kind == "frequency"
    np.testing.assert_allclose(exported.axes[0].ppm.min(), -50.0)
    np.testing.assert_allclose(exported.axes[0].ppm.max(), 150.0)


def test_main_window_simulate_csa_warns_when_the_pattern_is_clipped(qapp):
    """A powder pattern clipped by too narrow a ppm axis silently
    misrepresents the simulated lineshape -- this must be surfaced,
    not swallowed."""
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_diso_spin.setValue(50.0)
    window.sim_csa_aniso_spin.setValue(500.0)
    window.sim_csa_skeweta_spin.setValue(0.0)
    # Far too narrow for a 500 ppm anisotropy.
    window.sim_csa_ppm_min_spin.setValue(40.0)
    window.sim_csa_ppm_max_spin.setValue(60.0)

    window._on_simulate_csa_clicked()

    assert "clipped" in window.sim_csa_results_label.text()


def test_main_window_simulate_csa_finite_mas_needs_rate_and_larmor(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_regime_combo.setCurrentIndex(
        window.sim_csa_regime_combo.findData(False))
    window.sim_csa_mas_spin.setValue(0.0)
    window.sim_csa_larmor_spin.setValue(100.0)
    window._on_simulate_csa_clicked()
    assert warned and "spinning rate" in warned[-1][2]

    warned.clear()
    window.sim_csa_mas_spin.setValue(5000.0)
    window.sim_csa_larmor_spin.setValue(0.0)
    window._on_simulate_csa_clicked()
    assert warned and "Larmor" in warned[-1][2]


def test_main_window_simulate_csa_finite_mas_produces_an_exportable_spectrum(qapp):
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_regime_combo.setCurrentIndex(
        window.sim_csa_regime_combo.findData(False))
    window.sim_csa_mas_spin.setValue(5000.0)
    window.sim_csa_larmor_spin.setValue(100.0)
    window.sim_csa_diso_spin.setValue(50.0)
    window.sim_csa_aniso_spin.setValue(80.0)

    window._on_simulate_csa_clicked()

    assert window.sim_csa_export_button.isEnabled()
    assert "MAS" in window.sim_csa_results_label.text()
    assert "5000" in window.sim_csa_results_label.text()
    exported = window._simulated_csa_spectrum
    assert exported.metadata.sfo1 == pytest.approx(100.0)


def test_main_window_simulate_quadrupolar_refuses_an_unrecognised_nucleus(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_quad_nucleus_combo.setCurrentText("")
    window.sim_quad_larmor_spin.setValue(104.3)
    window._on_simulate_quadrupolar_clicked()

    assert warned, "no warning for a missing nucleus"
    assert not window.sim_quad_export_button.isEnabled()


def test_main_window_simulate_quadrupolar_first_order_produces_an_exportable_spectrum(qapp):
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_method_combo.setCurrentIndex(
        window.sim_quad_method_combo.findData("first_order"))
    window.sim_quad_spin_spin.setValue(2.5)
    window.sim_quad_cq_spin.setValue(3.0)
    window.sim_quad_eta_spin.setValue(0.2)
    window.sim_quad_larmor_spin.setValue(104.3)

    window._on_simulate_quadrupolar_clicked()

    assert window.sim_quad_export_button.isEnabled()
    assert "first-order" in window.sim_quad_results_label.text()

    before = len(window.session.spectra)
    window._on_export_simulated_quadrupolar_clicked()
    assert len(window.session.spectra) == before + 1
    exported = window.session.spectra[-1]
    assert exported.metadata.nucleus == "27Al"
    assert exported.metadata.sfo1 == pytest.approx(104.3)
    assert exported.data.shape == (window.sim_quad_points_spin.value(),)


def test_main_window_simulate_quadrupolar_exact_produces_a_different_result(qapp):
    """The whole point of the exact method: second-order quadrupolar
    effects the first-order eigenbasis calculation does not capture --
    at a low enough field / high enough C_Q the two must not coincide,
    or "Method" is not actually doing anything."""
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_spin_spin.setValue(2.5)
    window.sim_quad_cq_spin.setValue(8.0)
    window.sim_quad_eta_spin.setValue(0.3)
    window.sim_quad_larmor_spin.setValue(20.0)
    window.sim_quad_ppm_min_spin.setValue(-2000.0)
    window.sim_quad_ppm_max_spin.setValue(2000.0)

    window.sim_quad_method_combo.setCurrentIndex(
        window.sim_quad_method_combo.findData("first_order"))
    window._on_simulate_quadrupolar_clicked()
    first_order = np.asarray(window._simulated_quadrupolar_spectrum.data, copy=True)

    window.sim_quad_method_combo.setCurrentIndex(
        window.sim_quad_method_combo.findData("exact"))
    window._on_simulate_quadrupolar_clicked()
    exact = window._simulated_quadrupolar_spectrum.data

    assert "exact" in window.sim_quad_results_label.text()
    assert not np.allclose(first_order, exact)


def test_main_window_simulate_quadrupolar_mas_ct_is_fast_framed_and_reports_the_shift(qapp):
    """MAS quadrupolar (2026-09-25): a CT-only, second-order MAS pattern
    is simulated in well under a second, framed automatically, and the
    result line states the CT isotropic quadrupolar shift."""
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_cq_spin.setValue(5.0)
    window.sim_quad_eta_spin.setValue(0.2)
    window.sim_quad_larmor_spin.setValue(130.3)
    window.sim_quad_regime_combo.setCurrentIndex(window.sim_quad_regime_combo.findData("mas"))
    assert window.sim_quad_mas_spin.isEnabled()
    window.sim_quad_mas_spin.setValue(20000.0)
    window.sim_quad_transitions_combo.setCurrentIndex(
        window.sim_quad_transitions_combo.findData("ct"))
    window.sim_quad_method_combo.setCurrentIndex(
        window.sim_quad_method_combo.findData("second_order"))
    assert window.sim_quad_auto_axis_check.isChecked()

    t0 = time.perf_counter()
    window._on_simulate_quadrupolar_clicked()
    assert time.perf_counter() - t0 < 5.0

    text = window.sim_quad_results_label.text()
    assert "MAS 20 kHz" in text and "second-order" in text
    assert "CT isotropic quadrupolar shift" in text
    assert "clipped" not in text
    # framed on the CT: well inside +-200 ppm, not the default +-500
    assert -200 < window.sim_quad_ppm_min_spin.value() < 0 < window.sim_quad_ppm_max_spin.value() < 200
    sim = window._simulated_quadrupolar_spectrum.metadata.raw["simulation"]
    assert sim["regime"] == "mas" and sim["mas_Hz"] == 20000.0 and sim["transitions"] == "ct"
    # the CT centre of gravity sits at the isotropic quadrupolar shift
    data = window._simulated_quadrupolar_spectrum.data
    ppm = window._simulated_quadrupolar_spectrum.axes[0].ppm
    from opennmr.notebook import simulations as notebook_sims
    expected = notebook_sims.quadrupolar_ct_isotropic_shift_Hz(
        2.5, 5e6, 0.2, 130.3e6) / 130.3
    assert np.sum(ppm * data) / np.sum(data) == pytest.approx(expected, abs=0.05)


def test_main_window_simulate_quadrupolar_static_hides_the_mas_rate(qapp):
    window = MainWindow()
    window.sim_quad_regime_combo.setCurrentIndex(window.sim_quad_regime_combo.findData("static"))
    assert not window.sim_quad_mas_spin.isEnabled()


def test_main_window_simulate_quadrupolar_ct_only_refused_for_an_integer_spin(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_quad_nucleus_combo.setCurrentText("2H")
    window.sim_quad_cq_spin.setValue(0.16)
    window.sim_quad_larmor_spin.setValue(61.4)
    window.sim_quad_transitions_combo.setCurrentIndex(
        window.sim_quad_transitions_combo.findData("ct"))
    window._on_simulate_quadrupolar_clicked()
    assert warned and "central transition" in warned[0][2]
    assert not window.sim_quad_export_button.isEnabled()


def test_main_window_simulate_quadrupolar_compare_orders_draws_all_three(qapp):
    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_quad_nucleus_combo.setCurrentText("23Na")
    window.sim_quad_cq_spin.setValue(2.0)
    window.sim_quad_larmor_spin.setValue(105.8)
    window.sim_quad_transitions_combo.setCurrentIndex(
        window.sim_quad_transitions_combo.findData("ct"))
    window.sim_quad_compare_check.setChecked(True)
    window._on_simulate_quadrupolar_clicked()

    axes = window.simulation_canvas.figure.axes
    assert len(axes) == 2                                  # orders + differences
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert labels == ["First order", "Second order", "Exact"]
    assert window.canvas_tabs.currentWidget() is window.simulation_canvas
    # the exported spectrum is still the chosen method's (exact by default)
    assert window._simulated_quadrupolar_spectrum.metadata.raw["simulation"]["order"] == "exact"


def test_main_window_simulation_controls_are_seeded_from_the_selected_spectrum(qapp):
    """Same "seeded, never forced" pattern as the CSA/Dipolar fitting
    section's own MAS-rate seeding: a spectrum's field (from its SFO1 and
    its own nucleus) and nucleus fill the Simulate panels, but never
    overwrite a field or nucleus the user set.

    Since 2026-09-27 the panels take the field (T) and compute the Larmor
    frequency from it for their own nucleus (explicit user request), so
    what is seeded is the magnet's field, not a frequency: a 31P spectrum
    at 161.9 MHz puts a 23Na simulation on the same magnet at 23Na's
    frequency, not at 161.9 MHz."""
    from PySide6.QtTest import QTest
    from opennmr.notebook.utils import find_gamma
    window = MainWindow()
    axis = NMRAxis(kind="frequency", ppm=np.linspace(0, 10, 8))
    metadata = NMRMetadata(vendor="bruker", nucleus="31P", sfo1=161.9)
    spectrum = NMRSpectrum(data=np.zeros(8), axes=[axis], metadata=metadata)
    window._add_derived_spectrum(spectrum, "test")

    b0 = 161.9e6 / find_gamma("31P")["gamma"]
    assert window.sim_csa_nucleus_edit.text() == "31P"
    assert window.sim_csa_field_spin.value() == pytest.approx(b0, abs=1e-4)
    assert window.sim_csa_larmor_spin.value() == pytest.approx(161.9, abs=1e-3)
    assert window.sim_quad_field_spin.value() == pytest.approx(b0, abs=1e-4)
    # 31P is spin-1/2, so it has no quadrupole moment and must NOT seed
    # the quadrupolar simulator's nucleus.
    assert window.sim_quad_nucleus_combo.currentText() == ""

    # A field the user typed is not overwritten by a later selection; one
    # they left alone follows the new spectrum.
    # (The field only matters -- and is only editable -- for finite MAS.)
    window.sim_csa_regime_combo.setCurrentIndex(window.sim_csa_regime_combo.findData(False))
    field = window.sim_csa_field_spin
    field.lineEdit().selectAll()
    QTest.keyClicks(field.lineEdit(), "14.1")
    window.sim_csa_nucleus_edit.setText("1H")
    metadata2 = NMRMetadata(vendor="bruker", nucleus="23Na", sfo1=105.8)
    spectrum2 = NMRSpectrum(data=np.zeros(8), axes=[axis], metadata=metadata2)
    window._add_derived_spectrum(spectrum2, "test2")

    assert window.sim_csa_field_spin.value() == pytest.approx(14.1)
    assert window.sim_csa_nucleus_edit.text() == "1H"
    assert window.sim_quad_nucleus_combo.currentText() == "23Na"
    assert window.sim_quad_larmor_spin.value() == pytest.approx(105.8, abs=1e-3)


# ============================================================
# Accessibility: colour-blind/print-friendly palettes, trace colour
# presets, larger section titles, the Settings tab (interface font
# size, per-plot-type default styles)
# ============================================================

from opennmr.plotting import accessibility as plot_accessibility  # noqa: E402


def _select_palette_filter(window, key):
    window.output_palette_filter_combo.setCurrentIndex(
        window.output_palette_filter_combo.findData(key))


def test_palette_filter_lists_only_measured_friendly_maps(qapp):
    window = MainWindow()
    for key in ("colourblind", "print", "both"):
        _select_palette_filter(window, key)
        combo = window.output_colormap_combo
        listed = [combo.itemData(i) for i in range(combo.count())]
        assert listed == [name for name, _n in plot_accessibility.friendly_palettes(key)]
        # The text says how far each one can be trusted.
        assert "series)" in combo.itemText(0)
    _select_palette_filter(window, "all")
    combo = window.output_colormap_combo
    assert [combo.itemData(i) for i in range(combo.count())] == \
        plot_accessibility.series_palette_names()


def test_choosing_a_friendly_filter_makes_the_plots_friendly(qapp):
    window = MainWindow()
    window._select_series_colormap("RdBu")            # not greyscale-safe
    window._on_series_colormap_changed()
    assert all(c.style_settings()["compare_colormap"] == "RdBu" for c in window._all_canvases())

    _select_palette_filter(window, "print")
    chosen = window._series_colormap_name()
    assert chosen == plot_accessibility.friendly_palettes("print")[0][0]
    assert all(c.style_settings()["compare_colormap"] == chosen for c in window._all_canvases())


def test_friendly_filter_keeps_a_map_that_already_passes(qapp):
    window = MainWindow()
    assert window._series_colormap_name() == "viridis"
    assert "viridis" in dict(plot_accessibility.friendly_palettes("colourblind"))
    _select_palette_filter(window, "colourblind")
    assert window._series_colormap_name() == "viridis"


def test_a_map_hidden_by_the_filter_is_shown_by_resetting_it(qapp):
    window = MainWindow()
    _select_palette_filter(window, "print")
    window._select_series_colormap("RdBu")
    assert window.output_palette_filter_combo.currentData() == "all"
    assert window._series_colormap_name() == "RdBu"


def test_qualitative_accessible_palette_reaches_the_compare_plot(qapp):
    window = MainWindow()
    spectra = [_make_1d_peak_spectrum(x0=4.0 + 0.5 * i) for i in range(3)]
    window.compare_canvas.show_compare(spectra, mode="overlay")
    window._select_series_colormap("okabe_ito")
    window._on_series_colormap_changed()
    drawn = [matplotlib.colors.to_hex(e.colour) for e in window.compare_canvas.series()]
    expected = [matplotlib.colors.to_hex(c) for c in plot_accessibility.series_colours(3, "okabe_ito")]
    assert drawn == expected


def test_accessibility_line_checks_the_plot_on_screen(qapp):
    window = MainWindow()
    window.canvas_tabs.setTabVisible(window.canvas_tabs.indexOf(window.compare_canvas), True)
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    spectra = [_make_1d_peak_spectrum(x0=3.5 + 0.3 * i) for i in range(10)]
    window.compare_canvas.show_compare(spectra, mode="overlay")
    window._select_series_colormap("viridis")
    window._on_series_colormap_changed()
    text = window.series_accessibility_label.text()
    # Ten viridis traces are past what it can separate.
    assert text.startswith("⚠") and "10 series" in text

    window.compare_canvas.show_compare(spectra[:2], mode="overlay")
    window._refresh_slice_colour_rows()
    assert window.series_accessibility_label.text().startswith("✓")


def test_accessibility_line_describes_the_map_with_nothing_to_compare(qapp):
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window._refresh_series_accessibility_label()
    text = window.series_accessibility_label.text()
    n = plot_accessibility.max_safe_series("viridis", "colourblind")
    assert text.startswith("viridis:") and f"up to {n} series" in text


def test_trace_colour_presets_offer_both_sets_and_apply(qapp):
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    kinds = {kind for kind, _c in window._trace_preset_actions}
    assert kinds == {"colourblind", "print"}
    titles = [a.text() for a in window.trace_color_presets_menu.actions()]
    assert titles == ["Colour blind-friendly", "Print-friendly"]

    window._trace_preset_actions[("colourblind", "#0072B2")].trigger()
    assert window.canvas.style_settings()["trace_color"] == "#0072B2"
    assert window.trace_color_button.text() == "#0072B2"
    # Only the plot being styled, as the free colour picker does.
    assert window.fid_canvas.style_settings()["trace_color"] != "#0072B2"

    print_colour = plot_accessibility.trace_colour_presets("print")[1][1]
    window._trace_preset_actions[("print", print_colour)].trigger()
    assert window.canvas.style_settings()["trace_color"] == print_colour


def test_section_titles_are_bold_and_larger_than_their_text(qapp):
    window = MainWindow()
    processing_tab = window.side_tabs.widget(window._processing_tab_index)
    sections = {s.title(): s for s in processing_tab.findChildren(CollapsibleSection)}
    for title in ("FID Processing", "Phase", "Reference"):
        section = sections[title]
        header = section.headerFont()
        assert header.bold()
        assert header.pointSizeF() == pytest.approx(section.font().pointSizeF() * 1.25)


def test_settings_is_the_last_launcher_button(qapp):
    window = MainWindow()
    last = window.side_tabs.count() - 1
    assert window.side_tabs.tabText(last) == "Settings"
    assert window._settings_tab_index == last
    settings_tab = window.side_tabs.widget(last)
    titles = [s.title() for s in settings_tab.findChildren(CollapsibleSection)]
    assert titles == ["Interface", "Default plot styles"]
    assert settings_tab.isAncestorOf(window.ui_font_size_spin)


def test_settings_menu_action_opens_the_settings_tab(qapp):
    window = MainWindow()
    assert window.settings_menu_action.shortcut().toString() == "Ctrl+,"
    window.settings_menu_action.trigger()
    assert window.side_tabs.currentIndex() == window._settings_tab_index


def test_numbered_panel_shortcuts_are_unchanged_by_settings(qapp):
    window = MainWindow()
    window_menu = _menu_by_title(window, "Window")
    shortcuts = {a.text(): a.shortcut().toString() for a in window_menu.actions()}
    # Ctrl+n is the n-th toolbar button.
    assert [shortcuts[name] for name in ("Load/Save", "Processing", "Compare", "Simulations")] == \
        ["Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+5"]
    assert "Ctrl+," not in [v for k, v in shortcuts.items() if k != "Settings…"]


def test_ui_font_size_scales_the_interface_and_keeps_titles_ahead(qapp):
    window = MainWindow()
    window.ui_font_size_spin.setValue(15)
    qapp.processEvents()
    assert qapp.font().pointSizeF() == pytest.approx(15)
    assert window.output_colormap_combo.font().pointSizeF() == pytest.approx(15)
    section = window.side_tabs.widget(window._processing_tab_index).findChildren(CollapsibleSection)[0]
    assert section.headerFont().pointSizeF() == pytest.approx(15 * 1.25)


def test_ui_font_size_is_saved_and_restored(qapp):
    window = MainWindow()
    window.ui_font_size_spin.setValue(13)
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert json.load(handle)["ui_font_size"] == 13

    qapp.setFont(window._system_ui_font)          # as a fresh launch would start
    window2 = MainWindow()
    assert window2.ui_font_size_spin.value() == 13
    assert qapp.font().pointSizeF() == pytest.approx(13)


def test_ui_font_reset_goes_back_to_the_system_size(qapp):
    window = MainWindow()
    system = window._system_ui_font.pointSizeF()
    window.ui_font_size_spin.setValue(20)
    window.ui_font_reset_button.click()
    assert qapp.font().pointSizeF() == pytest.approx(system)
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert "ui_font_size" not in json.load(handle)


def test_corrupt_ui_settings_start_at_defaults(qapp):
    with open(MainWindow._UI_SETTINGS_PATH, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    window = MainWindow()
    assert window.auto_apply_type_styles_checkbox.isChecked()


def _make_2d_spectrum(n_f1=8, n_f2=64):
    x_ppm = np.linspace(0, 10, n_f2)
    y_ppm = np.linspace(90, 100, n_f1)
    data = np.exp(-((x_ppm - 5) ** 2)) * np.ones((n_f1, 1))
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                           sfo1=400.0, sfo1_indirect=100.0)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=y_ppm, hz=y_ppm * 100.0),
                                        NMRAxis(kind="frequency", ppm=x_ppm, hz=x_ppm * 400.0)],
                       metadata=metadata)


def _save_type_default_from_panel(window, key, **spins):
    for attr, value in spins.items():
        getattr(window, attr).setValue(value)
    window.settings_type_save_buttons[key].click()


def test_settings_tab_saves_a_default_per_plot_type(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window = MainWindow()
    assert window.settings_type_status_labels["2d"].text() == "not set"
    assert not window.settings_type_apply_buttons["2d"].isEnabled()
    window._set_trace_colour("#882255")
    _save_type_default_from_panel(window, "2d", output_fontsize_spin=17, output_contour_levels_spin=12)

    saved = window._load_figure_styles()["2d"]
    assert saved["labelsize"] == 17
    assert saved["contour_clevels"] == 12
    assert saved["trace_color"] == "#882255"      # the per-type defaults used to drop this
    assert saved["compare_colormap"] == "viridis"
    assert window.settings_type_status_labels["2d"].text() == "saved"
    assert window.settings_type_apply_buttons["2d"].isEnabled()

    window.settings_type_clear_buttons["2d"].click()
    assert "2d" not in window._load_figure_styles()
    assert window.settings_type_status_labels["2d"].text() == "not set"


def test_saved_type_default_is_used_when_that_kind_of_plot_is_drawn(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    window = MainWindow()
    window._save_figure_style("2d", {"labelsize": 19, "contour_clevels": 11, "linewidth": 2.0})

    window.canvas.show_spectrum(_make_1d_peak_spectrum())
    assert window.canvas.style_settings()["labelsize"] != 19

    window.canvas.show_spectrum(_make_2d_spectrum())
    settings = window.canvas.style_settings()
    assert settings["labelsize"] == 19
    assert settings["contour_clevels"] == 11
    assert window.canvas.ax.xaxis.label.get_fontsize() == pytest.approx(19)

    # A change made on the 2D plot survives showing another 2D: the
    # default is for a change of *kind*, not every redraw.
    window.canvas.set_style(labelsize=9)
    other = _make_2d_spectrum(n_f1=6, n_f2=48)
    window.canvas.show_spectrum(other)
    assert window.canvas.style_settings()["labelsize"] == 9
    # ... but a new 2D spectrum's contour levels reset to the saved
    # default, not the built-in 7.
    assert window.canvas.style_settings()["contour_clevels"] == 11


def test_type_defaults_can_be_switched_off(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    window = MainWindow()
    window._save_figure_style("2d", {"labelsize": 19})
    window.auto_apply_type_styles_checkbox.setChecked(False)
    window.canvas.show_spectrum(_make_2d_spectrum())
    assert window.canvas.style_settings()["labelsize"] != 19
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert json.load(handle)["auto_apply_type_styles"] is False

    # Apply still works on demand.
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.settings_type_apply_buttons["2d"].click()
    assert window.canvas.style_settings()["labelsize"] == 19
    assert window.output_fontsize_spin.value() == 19


def test_pseudo2d_stack_is_its_own_plot_type(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    window = MainWindow()
    window._save_figure_style("pseudo2d", {"compare_colormap": "tol_muted", "labelsize": 16})
    window._save_figure_style("compare", {"labelsize": 11})
    delays = [0.01, 0.05, 0.1, 0.2]
    window._show_pseudo2d_stack(_make_pseudo2d_relaxation_spectrum(delays))
    assert window.canvas.plot_kind() == "pseudo2d"
    assert window.canvas.style_settings()["labelsize"] == 16
    assert window.canvas.style_settings()["compare_colormap"] == "tol_muted"


# ============================================================
# Simulations: Diffusion (PFG) and Shaped pulse (Bloch) sections --
# ported from upstream (Leo, 1c5613e), adapted from his Simulate mode
# (toolbar button + sub-tabs) to two collapsible sections of the
# Simulations launcher tile.
# ============================================================

def _enter_simulations(window):
    window.side_tabs.setCurrentIndex(window._simulations_tab_index)


def test_simulations_tile_has_diffusion_and_bloch_sections(qapp):
    window = MainWindow()
    page = window.side_tabs.widget(window._simulations_tab_index)
    titles = [s.title() for s in page.findChildren(CollapsibleSection)]
    assert titles == ["J-coupling", "CSA", "Quadrupolar", "Czjzek distribution",
                      "Combined Hamiltonian", "Diffusion (PFG)", "Shaped pulse (Bloch)",
                      "pNMR (CIF)"]


def test_simulation_plot_shows_on_run_and_hides_on_leaving_the_tile(qapp):
    window = MainWindow()
    tab = window.canvas_tabs.indexOf(window.simulation_canvas)
    assert not window.canvas_tabs.isTabVisible(tab)
    _enter_simulations(window)
    # Entering alone does not steal the plot: CSA/Quadrupolar draw on
    # the spectrum canvases, so only running one of these raises it.
    assert not window.canvas_tabs.isTabVisible(tab)
    window._run_diffusion_simulation()
    assert window.canvas_tabs.isTabVisible(tab)
    assert window.canvas_tabs.currentWidget() is window.simulation_canvas
    assert window.simulation_canvas.figure.axes
    assert "b at max gradient" in window.sim_diffusion_summary_label.text()

    window.side_tabs.setCurrentIndex(0)
    assert window.canvas_tabs.currentWidget() is window.canvas
    assert not window.canvas_tabs.isTabVisible(tab)


def test_diffusion_simulation_compares_shapes_and_sequences(qapp):
    window = MainWindow()
    _enter_simulations(window)
    window.sim_d_edit.setText("2e-9")
    window.sim_compare_combo.setCurrentIndex(window.sim_compare_combo.findData("shape"))
    ax = window.simulation_canvas.figure.axes[0]
    assert len(ax.lines) == 5
    summary = window.sim_diffusion_summary_label.text()
    assert "Half-sine" in summary and "Rectangular" in summary
    assert window._sim_form.isRowVisible(window.sim_ramp_spin)  # comparing across it

    window.sim_compare_combo.setCurrentIndex(window.sim_compare_combo.findData("sequence"))
    assert len(window.simulation_canvas.figure.axes[0].lines) == 4

    window.sim_compare_combo.setCurrentIndex(0)
    window.sim_d_edit.setText("")
    window._run_diffusion_simulation()
    assert len(window.simulation_canvas.figure.axes[0].lines) == 8  # the leonmr decade sweep
    assert not window._sim_form.isRowVisible(window.sim_tau_spin)  # PGSE has no tau


def test_diffusion_simulation_matches_stejskal_tanner(qapp):
    # Independent physics check through the GUI path: rectangular PGSE,
    # I/I0 = exp(-gamma^2 G^2 delta^2 (Delta - delta/3) D) at max gradient.
    window = MainWindow()
    _enter_simulations(window)
    window.sim_nucleus_combo.setCurrentText("1H")
    window.sim_shape_combo.setCurrentIndex(window.sim_shape_combo.findData("rect"))
    window.sim_delta_spin.setValue(1.0)
    window.sim_DELTA_spin.setValue(20.0)
    window.sim_max_gradient_spin.setValue(0.5)
    window.sim_d_edit.setText("2e-9")
    window._run_diffusion_simulation()
    gamma = 267.52218744e6
    b = gamma**2 * 0.5**2 * 1e-3**2 * (20e-3 - 1e-3 / 3)
    expected = np.exp(-b * 2e-9)
    reported = float(window.sim_diffusion_summary_label.text().split("I/I₀ at max = ")[1].split()[0])
    assert reported == pytest.approx(expected, rel=2e-3)


def test_diffusion_simulation_reports_bad_input_instead_of_raising(qapp):
    window = MainWindow()
    _enter_simulations(window)
    window.sim_d_edit.setText("-1")
    window._run_diffusion_simulation()
    assert window.sim_diffusion_summary_label.text().startswith("Cannot simulate")


def test_bloch_simulation_gives_the_requested_flip(qapp):
    window = MainWindow()
    _enter_simulations(window)
    window.bloch_shape_combo.setCurrentIndex(window.bloch_shape_combo.findData("GAUSSIAN"))
    window.bloch_flip_spin.setValue(90.0)
    window.bloch_n_offsets_spin.setValue(51)
    window._run_bloch_simulation()
    text = window.bloch_summary_label.text()
    assert "Mz = 0.000" in text or "Mz = -0.000" in text
    assert len(window.simulation_canvas.figure.axes) == 3  # shape, its phase twin, profile
    assert window.canvas_tabs.currentWidget() is window.simulation_canvas

    # The BURP shapes offer the flip they were designed for.
    window.bloch_shape_combo.setCurrentIndex(window.bloch_shape_combo.findData("REBURP"))
    assert window.bloch_flip_spin.value() == 180.0

    window.bloch_power_mode_combo.setCurrentIndex(window.bloch_power_mode_combo.findData("hard"))
    assert window._bloch_form.isRowVisible(window.bloch_hard_width_spin)
    assert not window._bloch_form.isRowVisible(window.bloch_flip_spin)



# ============================================================
# Compare as its own launcher tile; the Open experiments list shared
# under the tiles on every tile but Simulate (explicit user request,
# 2026-09-23).
# ============================================================

def test_compare_is_its_own_tile_with_its_icon(qapp):
    window = MainWindow()
    tiles = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    assert tiles[window._compare_tab_index] == "Compare"
    # Third, after Process (toolbar order, user request 2026-09-27).
    assert tiles.index("Process") < tiles.index("Compare") < tiles.index("Fit")
    assert not window.side_tabs._buttons[window._compare_tab_index].icon().isNull()


def test_compare_icon_resource_is_a_usable_square_image(qapp):
    # The tile icons are edited in place in gui/resources (assets/ holds
    # design sources that need not match them), so check what matters:
    # the icon exists, loads, and is square like every other launcher icon.
    from PySide6.QtGui import QImage
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "src", "opennmr", "gui", "resources", "launcher_comparison.png")
    image = QImage(path)
    assert not image.isNull()
    assert image.width() == image.height() >= 64


def test_compare_controls_moved_out_of_open(qapp):
    window = MainWindow()
    open_page = window.side_tabs.widget(0)
    compare_page = window.side_tabs.widget(window._compare_tab_index)
    for widget in (window.compare_list, window.compare_status_label,
                   window.layout_mode_combo, window.compare_normalize_combo,
                   window.combine_op_combo):
        assert compare_page.isAncestorOf(widget)
        assert not open_page.isAncestorOf(widget)
    # Open keeps browsing and the selected-spectrum summary only.
    assert [s.title() for s in open_page.findChildren(CollapsibleSection)] == ["Selected spectrum"]
    assert open_page.isAncestorOf(window.data_browser)
    assert not open_page.isAncestorOf(window.spectrum_list)


def test_open_experiments_list_sits_under_the_tiles(qapp):
    window = MainWindow()
    assert window.open_experiments_box.isAncestorOf(window.spectrum_list)
    assert window.left_splitter.indexOf(window.side_tabs) == 0
    assert window.left_splitter.indexOf(window.open_experiments_box) == 1
    assert not window.side_tabs.isAncestorOf(window.spectrum_list)


def test_open_experiments_list_shows_on_every_tile_but_simulate(qapp):
    window = MainWindow()
    window.show()
    try:
        for index in range(window.side_tabs.count()):
            window.side_tabs.setCurrentIndex(index)
            qapp.processEvents()
            name = window.side_tabs.tabText(index)
            # Simulate and the Periodic Table work without open data.
            assert window.open_experiments_box.isVisible() == (
                name not in ("Simulate", "Periodic Table")), name
    finally:
        window.close()


def test_switching_experiment_from_another_tile_keeps_that_tile(qapp, tmp_path):
    # The point of the shared list: change experiment without going
    # back to Open.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._fitting_tab_index)
    window.spectrum_list.setCurrentRow(0)
    assert window._selected_index() == 0
    assert window.side_tabs.currentIndex() == window._fitting_tab_index


def test_multi_selecting_while_on_compare_fills_the_comparison(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._compare_tab_index)
    _select_rows(window, [0, 1])
    assert window.compare_list.count() == 2
    assert window.side_tabs.currentIndex() == window._compare_tab_index


def test_window_menu_compare_entry_opens_the_compare_tile(qapp):
    window = MainWindow()
    window_menu = _menu_by_title(window, "Window")
    next(a for a in window_menu.actions() if a.text() == "Compare").trigger()
    assert window.side_tabs.currentIndex() == window._compare_tab_index


def test_launcher_tiles_are_a_persistent_row_above_both_columns(qapp):
    """Explicit request: "the big buttons should be a persistent row
    along the top of the app". In the side panel they wrapped to three
    or four rows at any usable width and ate the panel's own height."""
    window = MainWindow()
    assert window.tile_bar is not None
    # Above the splitter in the shell, not inside the left column.
    shell = window.centralWidget()
    layout = shell.layout()
    assert layout.indexOf(window.tile_bar) < layout.indexOf(window.main_splitter)
    assert window.tile_bar.parent() is shell
    # It still switches panels.
    window.side_tabs.setCurrentIndex(2)
    assert window.side_tabs._buttons[2].isChecked()
    window.side_tabs._buttons[1].click()
    assert window.side_tabs.currentIndex() == 1


def test_appearance_setting_switches_theme_and_persists(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_UI_SETTINGS_PATH", str(tmp_path / "ui.json"))
    window = MainWindow()
    # Default is "follow system", not a forced light or dark.
    assert window.ui_theme_combo.currentData() == "system"

    dark_index = window.ui_theme_combo.findData("dark")
    window.ui_theme_combo.setCurrentIndex(dark_index)
    qapp.processEvents()
    assert window._ui_settings["ui_theme"] == "dark"
    assert (tmp_path / "ui.json").exists()

    # A fresh window picks the saved choice back up.
    reopened = MainWindow()
    assert reopened.ui_theme_combo.currentData() == "dark"

    window.ui_theme_combo.setCurrentIndex(window.ui_theme_combo.findData("system"))
    qapp.processEvents()


def test_quadrupolar_nucleus_is_an_editable_combo_that_fills_spin(qapp):
    """Explicit request: same kind of box as the PFG simulator's, and
    the spin read from the table so the two cannot disagree."""
    window = MainWindow()
    combo = window.sim_quad_nucleus_combo
    assert combo.isEditable()                       # like sim_nucleus_combo
    assert window.sim_nucleus_combo.isEditable()
    listed = [combo.itemText(i) for i in range(combo.count())]
    assert "27Al" in listed and "23Na" in listed
    # Spin-1/2 nuclides have no quadrupole moment, so they are not offered.
    assert "1H" not in listed and "13C" not in listed

    combo.setCurrentText("27Al")
    qapp.processEvents()
    assert window.sim_quad_spin_spin.value() == pytest.approx(2.5)
    combo.setCurrentText("2H")
    qapp.processEvents()
    assert window.sim_quad_spin_spin.value() == pytest.approx(1.0)
    combo.setCurrentText("23Na")
    qapp.processEvents()
    assert window.sim_quad_spin_spin.value() == pytest.approx(1.5)

    # Mid-typing is not a nuclide; it must not reset or warn.
    before = window.sim_quad_spin_spin.value()
    combo.setCurrentText("2")
    qapp.processEvents()
    assert window.sim_quad_spin_spin.value() == pytest.approx(before)

    # I is a property of the nuclide, not a free parameter: derived
    # from the box above and not typeable beside it, so the two cannot
    # be made to disagree by hand.
    assert window.sim_quad_spin_spin.isReadOnly()
    assert window.sim_quad_spin_spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
    # Read-only, not disabled -- the value must stay legible.
    assert window.sim_quad_spin_spin.isEnabled()
