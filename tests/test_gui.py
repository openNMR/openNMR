import dataclasses
import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QSplitter

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


def test_main_window_phase_adjustment_preserves_zoomed_view(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")

    # Simulate the user having zoomed in before touching the phase.
    window.canvas.ax.set_xlim(2.0, 4.0)
    window.canvas.ax.set_ylim(-0.5, 0.5)

    window.phase0_handle.setValue(45.0)

    assert window.canvas.ax.get_xlim() == (2.0, 4.0)
    assert window.canvas.ax.get_ylim() == (-0.5, 0.5)


def test_main_window_pivot_change_updates_preview(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    index = window._selected_index()

    window.phase1_spin.setValue(60.0)
    with_default_pivot = window.session.spectra[index].data.copy()

    window.pivot_spin.setValue(window.pivot_spin.value() + 1.0)
    with_shifted_pivot = window.session.spectra[index].data

    assert not np.allclose(with_default_pivot, with_shifted_pivot)


def test_main_window_phase_range_is_wide(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    window.phase1_spin.setValue(1000.0)
    assert window.phase1_spin.value() == pytest.approx(1000.0)


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


def _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="rawexpt1"):
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
    fid = _synthetic_raw_fid(n=n)
    dic = {"acqus": acqus}
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)
    return str(expt)


def _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=32, name="rawexpt2d", f1_nuc="13C"):
    from opennmr.io import bruker as bruker_io

    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13, "WDW": 0})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 100.6, "WDW": 0})
    (expt / "ser").write_bytes(b"\x00")
    (expt / "acqus").write_bytes(b"\x00")

    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    # `acqu2s` is read from the mocked `ng.bruker.read()` dict, not
    # parsed off disk -- matching real Bruker behaviour (nmrglue reads
    # every acqu*s/proc*s file itself as part of one `read()` call).
    acqu2s = {"NUC1": f1_nuc, "BF1": 100.6, "O1": 0.0}
    t2 = np.arange(n_f2) / 8000.0
    row = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000.0 * t2)
    fid = np.tile(row, (n_f1, 1)).astype(np.complex128)
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker_io.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker_io.ng.bruker, "remove_digital_filter", lambda d, data: data)
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


def test_main_window_reprocess_fid_with_window_override_changes_spectrum(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()
    original = window.session.spectra[index].data.copy()
    window._set_zero_fill_value(64)  # isolate the window change from the new default-zero-fill behaviour

    hann_row = next(i for i in range(window.window_combo.count()) if window.window_combo.itemData(i) == "hann")
    window.window_combo.setCurrentIndex(hann_row)
    window._on_reprocess_fid_clicked()

    updated = window.session.spectra[index].data
    assert len(updated) == 64
    assert not np.allclose(original, updated)
    assert window.session.spectra[index].metadata.raw.get("fid") is not None


def test_main_window_reprocess_fid_zero_fill_override_changes_length(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    index = window._selected_index()
    assert len(window.session.spectra[index].data) == 64

    window._set_zero_fill_value(128)
    window._on_reprocess_fid_clicked()

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
    assert "Magnitude" in labels

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
    assert "Magnitude" in labels


def test_main_window_fid_preview_toggle_shows_real_only(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    window.fid_display_combo.setCurrentIndex(1)  # Real
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Real" in labels
    assert "Magnitude" not in labels


def test_main_window_fid_preview_tracks_selected_slice(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)

    window.slice_spin.setValue(3)
    assert window.fid_canvas.ax.get_title() == "Slice 3"


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


def test_main_window_tdeff_spin_enabled_and_prefilled_for_xpulse(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    assert window.tdeff_spin.isEnabled() is True
    assert window.tdeff_spin.value() == 3  # prefilled from the file's own recorded TDEFF


def test_main_window_tdeff_spin_disabled_for_bruker(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    assert window.tdeff_spin.isEnabled() is False
    assert window.tdeff_spin.value() == 0


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


def test_main_window_reprocess_fid_works_for_varian(qapp, tmp_path):
    # Regression test: before _fid_reprocessable recognised "varian",
    # this control was simply hidden; now that it's shown, Reprocess
    # must actually succeed (varian.read_1d/read_2d take the same
    # fid_shift/align_echo/linear_predict kwargs the GUI always sends).
    window = MainWindow()
    window._load(_make_varian_dataset(tmp_path, n=8), vendor="varian")
    index = window._selected_index()

    window.fid_shift_spin.setValue(1.0)
    window._on_reprocess_fid_clicked()  # must not raise

    assert len(window.session.spectra[index].data) > 8  # zero-filled by the default preset


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
    window._on_reprocess_fid_clicked()

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

    window.align_echo_button.setChecked(True)
    aligned = window.session.spectra[index].data.copy()

    window.align_echo_button.setChecked(False)
    reverted = window.session.spectra[index].data

    assert not np.allclose(aligned, reverted)
    labels = [line.get_label() for line in window.fid_canvas.ax.lines]
    assert "Echo top" not in labels


def test_main_window_reprocess_updates_both_spectrum_and_fid_canvases(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    window.canvas_tabs.setCurrentWidget(window.fid_canvas)

    window.tdeff_spin.setValue(2)
    window._on_reprocess_fid_clicked()

    # FID tab was active at reprocess time -> refreshed in place, not
    # left showing the pre-reprocess plot (the TDeff marker moves to
    # the newly-set cutoff).
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
    assert len(window.compare_canvas.ax.lines) == 2
    assert window.canvas_tabs.currentWidget() is window.compare_canvas


def test_main_window_deselecting_back_to_single_restores_single_view(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert len(window.compare_canvas.ax.lines) == 2

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
    assert window.stack_gap_spin.isEnabled() is False
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
    plotted_a = window.compare_canvas.ax.lines[0].get_ydata()
    # Plotted line reflects the /mass=2.0 scaling...
    assert not np.allclose(original_a, plotted_a)
    np.testing.assert_allclose(plotted_a * 2.0, original_a)
    # ...but the session's own stored data is untouched (comparison
    # copies are ephemeral, never written back).
    np.testing.assert_allclose(original_a, window.session.spectra[0].data)


def test_main_window_set_mass_action_via_dialog(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")

    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (3.5, True))
    window._on_set_mass_action(0)

    assert window.session.spectra[0].metadata.mass == pytest.approx(3.5)
    assert "Sample mass" in window.info_label.text()


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
    # Ctrl-clicking one more spectrum in the Spectra tab shouldn't wipe
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
    window.peak_table.setRowCount(0)  # what the "Clear All" button's own click handler calls
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


def test_main_window_peak_picked_adds_row_only_when_pick_mode_active(qapp):
    window = MainWindow()
    window.canvas.peakPicked.emit(5.5)  # pick mode off -- should be ignored
    assert window.peak_table.rowCount() == 0

    window.pick_peak_button.setChecked(True)
    window.canvas.peakPicked.emit(5.5)
    assert window.peak_table.rowCount() == 1
    assert window.peak_table.item(0, 0).text() == "5.5"


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
    # Data + Fit (total) + one peak component + baseline, for the single peak fitted above.
    assert len(window.spectral_fit_canvas.ax.lines) == 4

    _load_spectrum_directly(window, spectrum_b)
    assert len(window.spectral_fit_canvas.ax.lines) == 0

    window._select_single_row(0)
    assert len(window.spectral_fit_canvas.ax.lines) == 4


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

    window._on_clear_clicked()

    assert window.peak_table.rowCount() == 0
    assert window.analysis_results_label.text() == ""
    assert "pseudo-2D" in window.analysis_status_label.text()


# ============================================================
# Zero-fill dropdown (standard sizes, default = 2x NP)
# ============================================================

def test_main_window_zero_fill_defaults_to_2x_np_rounded_to_preset(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import _ZERO_FILL_SIZES

    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=600), vendor="bruker")
    # 2*600 = 1200 -- the smallest standard preset at or above that is 2048 (2k).
    expected = next(s for s in _ZERO_FILL_SIZES if s >= 1200)
    assert window._zero_fill_value() == expected
    assert window.zero_fill_custom_spin.isHidden() is True


def test_main_window_zero_fill_falls_back_to_custom_above_largest_preset(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=200_000), vendor="bruker")
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
    window._on_reprocess_fid_clicked()

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
    window._on_reprocess_fid_clicked()

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
    window._on_reprocess_fid_clicked()

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
    window._on_reprocess_fid_clicked()

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

def test_main_window_has_five_canvas_tabs(qapp):
    window = MainWindow()
    labels = [window.canvas_tabs.tabText(i) for i in range(window.canvas_tabs.count())]
    assert labels == ["FID", "Spectrum", "Spectral Fit", "Fit", "Compare"]
    assert window.canvas_tabs.widget(0) is window.fid_canvas
    assert window.canvas_tabs.widget(1) is window.canvas
    # Fit/Compare start hidden -- nothing to show until a pseudo-2D
    # spectrum is selected, or the Compare list has an entry.
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.fit_canvas)) is False
    assert window.canvas_tabs.isTabVisible(window.canvas_tabs.indexOf(window.compare_canvas)) is False


def test_main_window_selecting_spectrum_switches_to_spectrum_tab(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert window.canvas_tabs.currentWidget() is window.compare_canvas

    window._select_single_row(0)
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_clear_clears_all_five_canvases(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    assert len(window.compare_canvas.ax.lines) == 2

    window._on_clear_clicked()

    for canvas in (
        window.canvas, window.fid_canvas, window.compare_canvas, window.fit_canvas, window.spectral_fit_canvas,
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

    assert len(window.canvas.ax.lines) == 5
    assert window.canvas_tabs.currentWidget() is window.canvas


def test_main_window_pseudo2d_view_resets_to_contour_on_reselection(qapp, tmp_path):
    window = MainWindow()
    window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=8, name="a"), vendor="bruker")
    window.pseudo2d_view_combo.setCurrentIndex(1)

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
    assert keys == [key for _label, key in _RELAXATION_MODEL_ITEMS]


def test_main_window_fit_type_diffusion_switches_model_list_and_label(qapp):
    from opennmr.gui.main_window import _DIFFUSION_MODEL_ITEMS

    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)  # Diffusion (PFG)

    keys = [window.relaxation_model_combo.itemData(i) for i in range(window.relaxation_model_combo.count())]
    assert keys == [key for _label, key in _DIFFUSION_MODEL_ITEMS]
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
    labels = [line.get_label() for line in window.spectral_fit_canvas.ax.lines]
    assert "Data" in labels and "Fit (total)" in labels
    assert any(label.startswith("Peak 1") for label in labels)
    assert "Baseline" in labels
    assert "Slice 3" in window.spectral_fit_canvas.ax.get_title()
    assert "Peak 1" in window.spectral_fit_params_label.text()


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
