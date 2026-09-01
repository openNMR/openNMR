import dataclasses
import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter,
)

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.session import Session
from opennmr.processing.fitting import gaussian


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


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
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.0, "SFO1": 400.13, "O1": 2000.0, "NS": 4})
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
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1})
    _write_params(expt / "acqu2s", {"NUC1": f1_nuc, "BF1": 400.0, "O1": 0.0})
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
    left_tabs = window.centralWidget().widget(0)
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
    # Selection triggers a plot; canvas should now have a line drawn.
    assert len(window.canvas.ax.lines) == 1


def test_main_window_process_indirect_button_hidden_for_1d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.process_indirect_button.isHidden() is True


# ============================================================
# Menu bar
# ============================================================

def test_main_window_has_file_edit_window_menus(qapp):
    window = MainWindow()
    titles = [action.text().replace("&", "") for action in window.menuBar().actions()]
    assert titles == ["File", "Edit", "Window", "Analysis", "Fitting"]


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


def test_main_window_menu_window_panel_actions_switch_side_tabs(qapp):
    window = MainWindow()
    window_menu = window.menuBar().actions()[2].menu()
    analysis_action = next(a for a in window_menu.actions() if a.text() == "Analysis")
    analysis_action.trigger()
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Analysis"


def test_main_window_spectrum_canvas_defaults_to_full_resolution_others_do_not(qapp):
    # Explicit user feedback: the main Spectrum tab was silently,
    # severely downsampled by default -- "only the spectrum in fid
    # processing should be [downsampled]... I need to have the fully
    # sampled spectrum in the spectrum tab." Every other canvas keeps
    # the fast/approximate default, which is the right tradeoff there
    # (explicitly named: FID processing).
    window = MainWindow()
    assert window.canvas._full_resolution is True
    assert window.processing_preview_canvas._full_resolution is False
    assert window.fid_canvas._full_resolution is False
    assert window.compare_canvas._full_resolution is False
    assert window.fit_canvas._full_resolution is False
    assert window.spectral_fit_canvas._full_resolution is False


def test_main_window_menu_window_view_actions_switch_canvas_tabs(qapp):
    window = MainWindow()
    window_menu = window.menuBar().actions()[2].menu()
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

    window.spectrum_list.setCurrentRow(0)
    window._on_remove_clicked()
    assert window.spectrum_list.count() == 1
    assert len(window.session) == 1


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
    assert window.pivot_spin.value() == pytest.approx(expected_centre)


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

    window.pivot_slider.setValue(0)  # left end of the slider
    assert window.pivot_spin.value() == pytest.approx(float(freq_ppm.max()))

    window.pivot_slider.setValue(1000)  # right end of the slider
    assert window.pivot_spin.value() == pytest.approx(float(freq_ppm.min()))


def test_main_window_pivot_spin_moves_pivot_slider(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    freq_ppm = window._phase_freq_ppm

    window.pivot_spin.setValue(float(freq_ppm.max()))
    assert window.pivot_slider.value() == 0

    window.pivot_spin.setValue(float(freq_ppm.min()))
    assert window.pivot_slider.value() == 1000


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


def test_main_window_baseline_correct_is_now_on_by_default(qapp, tmp_path):
    # Explicit user feedback: a well-phased spectrum should generally
    # land close to zero, which wasn't reliably happening while this
    # was off by default for every vendor but X-Pulse.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    assert window.baseline_correct_button.isChecked() is True
    assert window.session.spectra[0].metadata.raw.get("baseline_correct") is True


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
    assert window.baseline_correct_button.isChecked() is True  # untouched spectrum, own (now-on) default
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


def test_main_window_current_phase_label_shows_dash_with_nothing_selected(qapp):
    window = MainWindow()
    assert window.current_phase_label.text() == "Current phase: —"


def test_main_window_current_phase_label_updates_live_during_a_drag(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.current_phase_label.text() == "Current phase: 0.00° / 0.00°"

    window.phase0_handle.setValue(45.0)

    assert window.current_phase_label.text() == "Current phase: 45.00° / 0.00°"
    assert window.phase0_spin.value() == pytest.approx(45.0)  # the delta control is untouched by this


def test_main_window_current_phase_label_survives_reselection_unlike_the_delta_spin_box(qapp, tmp_path):
    # Explicit user feedback: Phase 0/Phase 1 resetting to 0 on
    # reselection read as the phase value itself being lost -- this
    # label is what actually shows the still-applied total, while the
    # spin boxes keep their own (deliberate, delta-from-here) behavior
    # unchanged.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="expt_a"), vendor="bruker")
    index_a = window._selected_index()
    window._load(_make_1d_dataset(tmp_path, name="expt_b"), vendor="bruker")

    window._select_single_row(index_a)
    window.phase0_handle.setValue(45.0)
    assert window.current_phase_label.text() == "Current phase: 45.00° / 0.00°"

    window._select_single_row(1)  # away to B
    assert window.current_phase_label.text() == "Current phase: 0.00° / 0.00°"  # B was never phased

    window._select_single_row(index_a)  # back to A
    assert window.phase0_spin.value() == pytest.approx(0.0)  # the delta control resets, by design
    assert window.current_phase_label.text() == "Current phase: 45.00° / 0.00°"  # but the real value is shown


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
    # Regression test: starting to phase a pseudo-2D spectrum straight
    # from its default contour view (the common case -- a fresh
    # selection always lands there) used to silently reset any ppm
    # zoom the user had set on the contour, because there's no
    # existing Line2D yet for update_data to refresh in place, so a
    # full show_spectrum rebuild ran -- which resets the view by
    # design. X (ppm) means the same thing on a contour and a slice
    # line view, so it should survive that one-time rebuild.
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    assert window.canvas._contour_spectrum is not None  # confirms we're really on the contour

    window.canvas.ax.set_xlim(2.0, 4.0)

    window.phase0_handle.setValue(10.0)

    assert window.canvas.ax.get_xlim() == pytest.approx((2.0, 4.0))


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


def test_main_window_autophase_slices_button_phases_independently(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    index = window._selected_index()

    # Distort every slice's phase first, so autophase has something
    # real to correct regardless of how well-phased the synthetic
    # fixture data already was.
    window.apply_all_slices_checkbox.setChecked(True)
    window.phase0_handle.setValue(65.0)
    distorted = _complex(window.session.spectra[index]).copy()

    window._on_autophase_slices_clicked()

    updated = _complex(window.session.spectra[index])
    assert not np.allclose(distorted, updated)
    # A pure phase correction never changes magnitude, per row.
    np.testing.assert_allclose(np.abs(updated), np.abs(distorted), atol=1e-6)
    # Autophase (maximizing each row's own real-part integral) should
    # improve on -- never worsen -- the distorted starting point.
    for row in range(updated.shape[0]):
        assert np.sum(np.real(updated[row])) >= np.sum(np.real(distorted[row])) - 1e-6


def test_main_window_autophase_slices_button_hidden_for_1d(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert window.autophase_slices_button.isHidden() is True


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


def test_main_window_pseudo2d_phase_sets_canvas_slice_title(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8), vendor="bruker")
    window.slice_spin.setValue(3)

    window.phase0_handle.setValue(10.0)

    assert window.canvas.ax.get_title() == "Slice 3"


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


def _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="rawexpt1", probe=None):
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

    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    if probe is not None:
        acqus["PROBHD"] = probe
    fid = _synthetic_raw_fid(n=n)
    dic = {"acqus": acqus}
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)
    return str(expt)


def _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=32, name="rawexpt2d", f1_nuc="13C", ns=1.0,
                          fn_mode=None, fid=None):
    from opennmr.io import bruker as bruker_io

    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13, "WDW": 0})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 100.6, "WDW": 0})
    (expt / "ser").write_bytes(b"\x00")
    (expt / "acqus").write_bytes(b"\x00")

    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": ns}
    # `acqu2s` is read from the mocked `ng.bruker.read()` dict, not
    # parsed off disk -- matching real Bruker behaviour (nmrglue reads
    # every acqu*s/proc*s file itself as part of one `read()` call).
    acqu2s = {"NUC1": f1_nuc, "BF1": 100.6, "O1": 0.0}
    if fn_mode is not None:
        acqu2s["FnMODE"] = fn_mode
    if fid is None:
        t2 = np.arange(n_f2) / 8000.0
        row = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000.0 * t2)
        fid = np.tile(row, (n_f1, 1)).astype(np.complex128)
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker_io.ng.bruker, "read_acqus_file", lambda path: dic)
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)
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
            return label

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


def test_main_window_phase_spin_box_wide_enough_for_large_values(qapp):
    # Explicit user feedback: the Phase 0/1 spin boxes were too narrow
    # to read a value like "-1234.56" -- especially relevant now that
    # their range is unbounded (see _make_phase_control's own comment).
    window = MainWindow()
    for spin in (window.phase0_spin, window.phase1_spin):
        assert spin.minimumWidth() >= 80


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
    assert len(spec_ax.lines) == 1


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
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "off", "BF1": 100.6, "O1": 0.0}
    t2 = np.arange(n_f2) / 8000.0
    # A different decay rate per row -- slices are visibly distinct.
    fid = np.stack([np.exp(-t2 * 300 * (row + 1)) * np.exp(2j * np.pi * 1000.0 * t2) for row in range(n_f1)])
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker_io.ng.bruker, "read_acqus_file", lambda path: dic)
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)
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


def _make_xpulse_ntuples_2d_dataset(tmp_path, n1=4, n2=32, name="hsqc.jdx"):
    """A minimal synthetic NTUPLES 2D fixture -- deliberately
    asymmetric (n1 << n2, matching the real HSQC test dataset's own
    128x2048 shape) since that's exactly the shape that exposed the
    real fft_spectrum bug (len() reading axis 0 instead of the last
    axis for a 2D array) as a loud crash rather than a silent
    wrong-value bug a square array wouldn't have caught."""
    t2 = np.arange(n2) / 800.0
    f2signal = np.exp(-t2 * 50.0)
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
    for k in range(n1):
        sign = -1.0 if k % 2 else 1.0
        for kind, row in (("R", sign * f2signal), ("I", sign * f2signal * 0.5)):
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
    assert spectrum.data.shape[0] == 4  # F1: row-paired 8 -> decoded 4
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
    assert window.session.spectra[index].data.shape[0] == 4


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


def test_main_window_tdeff_spin_disabled_for_bruker(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
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
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_raw_fid(n=64)
    dic = {"acqus": acqus}
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)

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

    def fake_read(_path, **_kw):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(bruker_io.ng.bruker, "read", fake_read)
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


def test_main_window_f1_truncate_and_zero_fill_change_row_count_live_xpulse(qapp, tmp_path):
    window = MainWindow()
    path = _make_xpulse_ntuples_2d_dataset(tmp_path, n1=8, n2=16, name="f1livexpulse.jdx")
    window._load(path, vendor="xpulse")
    index = window._selected_index()
    assert window.session.spectra[index].data.shape[0] == 8

    window.tdeff_f1_spin.setValue(3)
    window.zero_fill_f1_spin.setValue(3)
    window._on_fid_processing_control_changed()
    _settle_live_reprocess(window)  # conventional 2D always debounces the actual recompute

    spectrum = window.session.spectra[index]
    assert spectrum.data.shape[0] == 3
    assert len(spectrum.axes[0].ppm) == 3


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

    # Reset reverts to _original_load_kwargs, snapshotted right after
    # _load's own "default every fresh 2D to pseudo-2D" step -- i.e.
    # *before* this test opted into genuine 2D above -- so Reset here
    # correctly lands back on pseudo-2D, not just "genuine 2D with no
    # truncate/zero-fill." Either way, F1's own controls must no
    # longer be reporting a stale truncated/zero-filled value.
    assert window.session.spectra[index].is_pseudo_2d is True
    assert window.session.spectra[index].data.shape[0] == 8
    assert window.tdeff_f1_spin.isEnabled() is False
    assert window.zero_fill_f1_spin.isEnabled() is False


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

    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    t2 = np.arange(n_f2)
    rows = []
    for row_index in range(n_f1):
        peak_at = 5 + row_index  # a different "echo top" every row -- no real shared echo
        row = np.zeros(n_f2, dtype=np.complex128)
        row[peak_at] = 1.0 + 0.5j
        row += 0.01 * np.exp(-((t2 - peak_at) ** 2) / 8.0)
        rows.append(row)
    fid = np.array(rows)
    dic = {"acqus": acqus, "acqu2s": {"NUC1": "off", "BF1": 400.0, "O1": 0.0}}
    monkeypatch.setattr(bruker_io.ng.bruker, "read_acqus_file", lambda path: dic)
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)
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
    assert len(window.canvas.ax.lines) == 1


def test_main_window_multi_selection_shows_compare(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.compare_status_label.text() == "Comparing 2 spectra."
    assert _compare_line_count(window.compare_canvas) == 2
    assert window.canvas_tabs.currentWidget() is window.compare_canvas


def test_main_window_compare_list_has_a_current_row_after_populating(qapp, tmp_path):
    # Explicit user feedback: "it is no longer obvious which spectrum
    # is selected" -- nothing ever set an initial current/highlighted
    # row in this list, only Move Up/Down did (and only after a click
    # had already set one), so a freshly (re)populated list showed no
    # current-row highlight at all.
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.compare_list.currentRow() == 0

    # Still current after Move Down shuffles the order, not silently
    # reset to row 0 again.
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
    assert window.compare_list.currentRow() == 0
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
    assert len(window.canvas.ax.lines) == 1
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_compare_skips_non_1d_spectra(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_2d_dataset(tmp_path, name="b2d"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert "not 1D" in window.compare_status_label.text()
    assert "Select at least two" in window.compare_status_label.text()


def test_main_window_compare_superimposed_layout_runs(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])

    window.layout_mode_combo.setCurrentIndex(1)  # Superimposed
    assert len(window.compare_canvas.ax.lines) == 2


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

    # Ctrl-click "a" too (add to selection, not replace it) so it
    # becomes current -- a plain setCurrentRow() would itself perturb
    # the ExtendedSelection state (the same Qt quirk _select_single_row
    # works around elsewhere), which isn't what this test is about.
    window.spectrum_list.item(0).setSelected(True)
    window.spectrum_list.setCurrentItem(window.spectrum_list.item(0))

    window._on_remove_clicked()

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
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"

    window._on_fit_curve_compare_stack_action()

    assert "T2" in window.analysis_results_label.text()
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
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    custom_index = window.relaxation_model_combo.findData("custom")
    window.relaxation_model_combo.setCurrentIndex(custom_index)
    window.custom_equation_edit.setText("A*exp(-x/T2)")
    window.custom_p0_edit.setText("1.0, 0.03")

    window._on_fit_curve_button_clicked()

    result = window._relaxation_fit_results[0]
    assert result["param_names"] == ["A", "T2"]
    assert result["params"]["T2"] == pytest.approx(true_t2, rel=0.1)
    assert "A=" in window.analysis_results_label.text()
    assert "T2=" in window.analysis_results_label.text()


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


def _load_spectrum_directly(window, spectrum):
    """Append an already-constructed NMRSpectrum straight into the
    session, bypassing file I/O -- for synthetic data no importer
    produces."""
    window.session.spectra.append(spectrum)
    window.session._load_info.append({"path": "synthetic", "vendor": "bruker", "kwargs": {}})
    window.spectrum_list.addItem(window.session.label(len(window.session) - 1))
    window._select_single_row(len(window.session) - 1)


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
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window.pick_peak_button.setChecked(True)
    window._on_peak_picked(5.0)
    assert len(window.canvas._peak_marker_lines) == 1


def test_main_window_removing_a_peak_removes_its_marker(qapp):
    window = MainWindow()
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
    _load_spectrum_directly(window, _make_1d_peak_spectrum(x0=5.0, fwhm=0.05))
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.5)
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([5.0])

    window.peak_table.item(0, 0).setText("6.5")
    assert window.canvas._peak_marker_lines[0].get_xdata() == pytest.approx([6.5])


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
    # the default prominence (0.1) should drop the weak one as noise,
    # while a lower one picks up both.
    ppm = np.linspace(3.0, 8.0, 2000)
    data = gaussian(ppm, 4.0, 0.05, 1.0) + gaussian(ppm, 6.5, 0.05, 0.08)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    spectrum = NMRSpectrum(data=data, axes=[axis], metadata=metadata)

    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window.auto_pick_prominence_spin.value() == pytest.approx(0.1)  # the raised default
    window._on_auto_pick_peaks_clicked()
    assert window.peak_table.rowCount() == 1  # weak peak dropped at the default

    window.auto_pick_prominence_spin.setValue(0.03)
    window._on_auto_pick_peaks_clicked()
    assert window.peak_table.rowCount() == 2  # both found at a lower threshold


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
    assert window.value_source_combo.currentData() == "integral"

    window.delay_source_combo.setCurrentIndex(1)
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
    window._add_peak_row(shift=5.0, fwhm=0.1, eta=0.5, fixed=True, bounds={"shift": (4.5, 5.5)})
    window.sn_noise_region_edit.setText("8, 9")
    window.sn_threshold_spin.setValue(4.0)
    window._save_current_analysis_state()

    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")  # switch away
    window._select_single_row(0)  # switch back -- triggers _restore_analysis_state

    assert window.peak_table.item(0, 3).checkState() == Qt.CheckState.Checked
    assert window.peak_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == {"shift": (4.5, 5.5)}
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
    window.manual_delays_edit.setText("1, 2, 3")
    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [1.0, 2.0, 3.0])


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
    window.delay_start_spin.setValue(0.0)
    window.delay_step_spin.setValue(0.25)
    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [0.0, 0.25, 0.5])


def test_main_window_resolve_delays_vclist_times_factor(qapp, tmp_path):
    expt = tmp_path / "expt"
    expt.mkdir()
    (expt / "vclist").write_text("1\n2\n4\n8\n")

    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1])
    spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, source_path=str(expt),
    ))
    window.delay_source_combo.setCurrentIndex(3)  # vclist x factor
    window.vclist_factor_spin.setValue(0.5)

    delays = window._resolve_independent_values(spectrum, is_diffusion=False)
    np.testing.assert_allclose(delays, [0.5, 1.0, 2.0, 4.0])


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
        return label

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
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"

    _fit_spectrum_then_curve(window)

    assert "T2" in window.analysis_results_label.text()
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

def test_main_window_peak_picking_and_integration_are_separate_subtabs(qapp):
    # Explicit user feedback: the combined "Spectral Fitting" sub-tab
    # held peak-picking, integration, and fit-bounds tuning all at
    # once -- "too much information at once for casual users". Peak
    # Picking and Integration (alternative "step 1"s, per their own
    # tooltips) are now separate sub-tabs.
    window = MainWindow()
    labels = [window.analysis_sub_tabs.tabText(i) for i in range(window.analysis_sub_tabs.count())]
    assert labels == ["Peak Picking", "Integration", "Curve Fitting", "VOCS", "Imaging"]

    peak_picking_tab = window.analysis_sub_tabs.widget(0)
    integration_tab = window.analysis_sub_tabs.widget(1)
    assert peak_picking_tab.isAncestorOf(window.peak_table)
    assert integration_tab.isAncestorOf(window.region_table)
    assert not peak_picking_tab.isAncestorOf(window.region_table)
    assert not integration_tab.isAncestorOf(window.peak_table)


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
    splitter = window.centralWidget()
    assert isinstance(splitter, QSplitter)
    tabs = splitter.widget(0)
    assert tabs.minimumWidth() >= 250


# ============================================================
# Multi-canvas plot tabs (Spectrum / FID / Compare / Fit)
# ============================================================

def test_main_window_has_six_canvas_tabs(qapp):
    window = MainWindow()
    labels = [window.canvas_tabs.tabText(i) for i in range(window.canvas_tabs.count())]
    assert labels == ["FID", "Processing Preview", "Spectrum", "Spectral Fit", "Fit", "Compare"]
    assert window.canvas_tabs.widget(0) is window.fid_canvas
    assert window.canvas_tabs.widget(1) is window.processing_preview_canvas
    assert window.canvas_tabs.widget(2) is window.canvas
    # Fit/Compare/Processing Preview start hidden -- nothing to show
    # until a pseudo-2D spectrum is selected, the Compare list has an
    # entry, or a FID-reprocessable spectrum is selected, respectively.
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.fit_canvas)) is False
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.compare_canvas)) is False
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.processing_preview_canvas)) is False


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

    assert "D" in window.analysis_results_label.text()
    assert "m^2/s" in window.analysis_results_label.text() or "e-" in window.analysis_results_label.text()
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

    results_text = window.analysis_results_label.text()
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
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)
    _fit_spectrum_then_curve(window)
    assert window.analysis_results_label.text() != ""

    window._select_single_row(1)
    assert window.analysis_results_label.text() == ""  # never fitted -- not spectrum a's result
    assert window.view_slice_fit_spin.isEnabled() is False

    window._select_single_row(0)
    assert window.analysis_results_label.text() != ""
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
    assert "Peak 1" in window.spectral_fit_params_label.text()

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
    # The two now live in separate Analysis sub-tabs (Spectral Fitting /
    # Curve Fitting), so search the whole sub-tab widget rather than one
    # page's own children.
    window = MainWindow()
    labels = [
        child.text() for child in window.analysis_sub_tabs.findChildren(QLabel) if child.text() in
        ("Spectrum fit (slice #):", "View curve fit (peak #):")
    ]
    assert set(labels) == {"Spectrum fit (slice #):", "View curve fit (peak #):"}


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
    assert window.canvas.ax.get_xlabel() == "Frequency / Hz"


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

def _make_1d_peak_spectrum_with_pulse(x0=5.0, fwhm=0.05, amplitude=1.0, n_pts=400, p1=5.0, sfo1=400.0):
    ppm = np.linspace(3.0, 7.0, n_pts)
    data = gaussian(ppm, x0, fwhm, amplitude)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1, raw={"acqus": {"P1": p1}})
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

def test_main_window_output_style_change_applies_to_every_canvas(qapp):
    window = MainWindow()
    window.output_linewidth_spin.setValue(3.0)
    for canvas in window._all_canvases():
        assert canvas._AXIS_LINEWIDTH == pytest.approx(3.0)


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


def test_main_window_output_compare_colormap_applies_to_all_canvases(qapp):
    window = MainWindow()
    window.output_colormap_combo.setCurrentText("plasma")
    for canvas in window._all_canvases():
        assert canvas._compare_colormap == "plasma"


def test_main_window_output_export_settings_stored_not_applied_live(qapp):
    # Export DPI/figure size are stored for File > Save Figure As...
    # only -- must not touch the live canvas figure at all (see
    # MplCanvas.set_export_settings's own docstring for why).
    window = MainWindow()
    original_dpi = window.canvas.figure.get_dpi()
    original_size = window.canvas.figure.get_size_inches()

    window.output_dpi_spin.setValue(600)
    window.output_fig_width_spin.setValue(10.0)
    window.output_fig_height_spin.setValue(7.0)

    assert window.canvas.figure.get_dpi() == pytest.approx(original_dpi)
    assert window.canvas.figure.get_size_inches() == pytest.approx(original_size)
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
    assert config == {"5 mm BBO": {"max_gradient_g_per_cm": pytest.approx(42.5)}}


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
