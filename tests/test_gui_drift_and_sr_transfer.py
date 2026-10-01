"""Process tab additions of 2026-10-01: the SR-from-another-nucleus
calculator in Reference, and Field Drift Correction (Najbauer &
Andreas, J. Magn. Reson. 305 (2019) 1-4)."""

import os

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import drift_correction as dc

from test_gui import _load_spectrum_directly, _make_raw_2d_dataset, _write_params, qapp  # noqa: F401
from test_drift_correction import _bruker_3d

SW1, SW2, N1, N2 = 2000.0, 8000.0, 16, 256
BF_H, BF_X = 400.13, 100.6


def _states_fid(start_hz, end_hz):
    """Bruker-convention States rows for one peak, drifting so that the
    correction is ``start_hz`` -> ``end_hz`` (see test_drift_correction)."""
    t2 = np.arange(N2) / SW2
    deltas = dc.drift_profile(N1, start_hz, end_hz)
    rows = []
    for k in range(N1):
        d = deltas[k]
        f2 = np.exp(-t2 * 60) * np.exp(2j * np.pi * (1200.0 - d) * t2)
        t1 = k / SW1
        f1 = 400.0 - d * BF_X / BF_H
        decay = np.exp(-t1 * 40)
        rows += [decay * np.cos(2 * np.pi * f1 * t1) * f2, decay * np.sin(2 * np.pi * f1 * t1) * f2]
    return np.array(rows)


def _corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def _genuine_2d(tmp_path, monkeypatch, name, fid):
    """_make_raw_2d_dataset, marked as TopSpin-processed in F1 so the
    window opens it as a genuine 2D rather than a series."""
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=N1, n_f2=N2, name=name, fn_mode=4, fid=fid)
    _write_params(os.path.join(path, "acqu2s"), {"NUC1": "13C", "BF1": BF_X})
    _write_params(os.path.join(path, "pdata", "1", "proc2s"),
                  {"SW_p": SW1, "SI": N1, "SF": BF_X, "WDW": 0, "FT_mod": 4})
    return path


def _clean_data(tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_genuine_2d(tmp_path, monkeypatch, "clean", _states_fid(0.0, 0.0)), vendor="bruker")
    assert not window.session.spectra[window._selected_index()].is_pseudo_2d
    return window.session.spectra[window._selected_index()].data.copy()


def _drifted_window(tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_genuine_2d(tmp_path, monkeypatch, "drifted", _states_fid(1.0, 40.0)), vendor="bruker")
    return window


# ----------------------------------------------------------------------
# Field drift correction
# ----------------------------------------------------------------------

def test_drift_section_is_in_process_and_cites_the_paper(qapp):  # noqa: F811
    window = MainWindow()
    window._show_side_tab("Process")
    assert window.drift_section.isVisibleTo(window.side_tabs.currentWidget())
    text = window.drift_panel.citation_label.text()
    assert "Najbauer" in text and "10.1016/j.jmr.2019.05.005" in text


def test_apply_and_remove_drift_correction_on_a_raw_2d(qapp, tmp_path, monkeypatch):  # noqa: F811
    clean = _clean_data(tmp_path, monkeypatch)
    window = _drifted_window(tmp_path, monkeypatch)
    index = window._selected_index()
    drifted = window.session.spectra[index].data.copy()
    assert _corr(drifted, clean) < 0.9

    panel = window.drift_panel
    panel.refresh()
    assert panel.apply_button.isEnabled() and not panel.remove_button.isEnabled()
    panel.start_spin.setValue(1.0)
    panel.end_spin.setValue(40.0)
    panel.apply_button.click()
    spectrum = window.session.spectra[index]
    assert _corr(spectrum.data, clean) > 0.999
    assert spectrum.metadata.raw["drift_correction"]["end_hz"] == 40.0
    assert window.session.get_load_kwargs(index)["drift_correction"] == {"start_hz": 1.0, "end_hz": 40.0}
    assert "Corrected" in panel.target_label.text()
    assert panel.remove_button.isEnabled()

    # An in-memory reprocess (what a phase drag or window change runs)
    # starts from the corrected FIDs, not the file's.
    window._recompute_multi_row_data(spectrum)
    assert _corr(window.session.spectra[index].data, clean) > 0.999

    # A further reload (another processing change) does not correct twice.
    window.session.reload(index)
    assert _corr(window.session.spectra[index].data, clean) > 0.999

    panel.remove_button.click()
    restored = window.session.spectra[index]
    assert "drift_correction" not in restored.metadata.raw
    assert _corr(restored.data, drifted) > 0.9999


def test_drift_correction_keeps_the_users_sr(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _drifted_window(tmp_path, monkeypatch)
    index = window._selected_index()
    window.sr_shift_spin.setValue(25.0)
    window._on_apply_sr_clicked()
    ppm_before = window.session.spectra[index].axes[-1].ppm.copy()
    window.drift_panel.start_spin.setValue(1.0)
    window.drift_panel.end_spin.setValue(40.0)
    window.drift_panel.apply_button.click()
    spectrum = window.session.spectra[index]
    assert spectrum.metadata.raw["sr_shift_hz"] == 25.0
    np.testing.assert_allclose(spectrum.axes[-1].ppm, ppm_before)
    assert window.sr_shift_spin.value() == 25.0


def test_saved_drift_correction_is_read_back_by_the_reader(qapp):  # noqa: F811
    kwargs = MainWindow._saved_processing_read_kwargs(
        {"drift_correction": {"start_hz": 1.0, "end_hz": 2.0, "endpoint": True, "n_points": 8,
                              "description": "x", "citation": "y"}})
    assert kwargs["drift_correction"] == {"start_hz": 1.0, "end_hz": 2.0, "endpoint": True}


def test_drift_correction_is_named_in_the_processing_log():
    from opennmr.io import history
    text = history.format_value("drift_correction", {"start_hz": 0.0, "end_hz": 12.5, "n_points": 64})
    assert "+12.5000 Hz" in text and "Najbauer" in text


def test_panel_refuses_a_1d_with_the_reason(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _one_d(4.7))
    window.drift_panel.refresh()
    assert not window.drift_panel.apply_button.isEnabled()
    assert "3D/4D" in window.drift_panel.target_label.text()
    assert window._apply_drift_correction_to_selected({"start_hz": 1.0, "end_hz": 2.0})


def _one_d(centre_ppm, nucleus="1H", sfo1=400.13, sr_hz=0.0):
    ppm = np.linspace(10, 0, 4096)
    data = 1.0 / (1 + ((ppm - centre_ppm) / 0.02) ** 2)
    raw = {"sr_recorded_hz": sr_hz} if sr_hz else {}
    meta = NMRMetadata(vendor="derived", nucleus=nucleus, sfo1=sfo1, bf1=sfo1, raw=raw)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)], metadata=meta)


def test_measure_fills_start_and_end_shifts(qapp):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _one_d(4.70))                    # before
    _load_spectrum_directly(window, _one_d(4.70 - 6.0 / 400.13))     # after: 6 Hz lower
    panel = window.drift_panel
    panel.refresh()
    panel.before_combo.setCurrentIndex(0)
    panel.after_combo.setCurrentIndex(1)
    panel.region_hi_spin.setValue(5.2)
    panel.region_lo_spin.setValue(4.2)
    panel.measure_button.click()
    assert panel.start_spin.value() == pytest.approx(0.0, abs=1e-3)
    assert panel.end_spin.value() == pytest.approx(6.0, abs=0.05)
    assert "drift of +6" in panel.measure_label.text()


def test_write_corrected_dataset_from_the_panel(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    src = str(tmp_path / "proj" / "5")
    _bruker_3d(src)
    window.drift_panel.start_spin.setValue(0.5)
    window.drift_panel.end_spin.setValue(4.0)
    result = window.drift_panel.write_dataset(src)
    assert os.path.isfile(os.path.join(result.destination, "ser"))
    assert result.destination.endswith(os.sep + "6")
    assert "Written to" in window.drift_panel.status_label.text()


def test_nus_dialog_suggests_a_new_expno(qapp, tmp_path):  # noqa: F811
    from opennmr.gui.drift_panel import NusConcatDialog
    a = tmp_path / "p" / "20"
    a.mkdir(parents=True)
    dialog = NusConcatDialog()
    dialog.add_source(str(a))
    dialog.add_source(str(tmp_path / "p" / "21"))
    assert dialog.sources() == [str(a), str(tmp_path / "p" / "21")]
    assert dialog.destination().endswith(os.sep + "21")  # first free beside the first block


# ----------------------------------------------------------------------
# SR from another nucleus
# ----------------------------------------------------------------------

def test_sr_transfer_from_an_open_experiment(qapp):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _one_d(4.7, "1H", 400.1326, sr_hz=-35.2))   # calibrated
    _load_spectrum_directly(window, _one_d(30.0, "13C", 100.6228))              # target, selected
    window.sr_transfer_section.setExpanded(True)
    combo = window.sr_cal_source_combo
    combo.setCurrentIndex(combo.findData(0))
    combo.activated.emit(combo.currentIndex())
    assert window.sr_cal_spin.value() == pytest.approx(-35.2)
    assert window.sfo1_cal_spin.value() == pytest.approx(400.1326)
    assert window.sfo1_new_spin.value() == pytest.approx(100.6228)
    expected = -35.2 / 400.1326 * 100.6228
    assert window._sr_transfer_result.sr_new_hz == pytest.approx(expected)
    assert f"{expected:.4f}" in window.sr_transfer_result_label.text()
    assert "same magnet" in window.sr_transfer_warning_label.text()

    window.sr_transfer_copy_button.click()
    assert QApplication.clipboard().text() == f"{expected:.4f}"

    window.sr_transfer_apply_button.click()
    target = window.session.spectra[1]
    assert target.metadata.raw["sr_shift_hz"] == pytest.approx(expected)
    assert window.sr_shift_spin.value() == pytest.approx(expected, abs=1e-4)


def test_sr_transfer_by_hand_and_a_mismatch_warning(qapp):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _one_d(30.0, "13C", 100.6228))
    window.sr_transfer_section.setExpanded(True)
    window.sr_cal_spin.setValue(10.0)
    window.sfo1_cal_spin.setValue(500.0)  # typed by hand
    assert window._sr_transfer_result.sr_new_hz == pytest.approx(10.0 / 500.0 * 100.6228)
    window.sfo1_cal_spin.setValue(0.0)
    assert window._sr_transfer_result is None
    assert not window.sr_transfer_apply_button.isEnabled()
