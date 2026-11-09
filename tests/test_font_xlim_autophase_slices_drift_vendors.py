"""2026-10-20 round: items 4, 5 and 6.

4. No Qt "missing font family" lookups for matplotlib-only faces, no
   matplotlib "identical xlims" warning for a one-point spectrum, and the
   harmless macOS font-weight message is filtered.
5. Field-drift correction for pseudo-2D series from every vendor, not only
   Bruker.
6. Autophase each slice of a pseudo-2D independently, as an option of its
   own.
"""

import logging
import warnings

import numpy as np
import pytest
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from opennmr.core import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import drift_correction as dc
from opennmr.notebook import slice_phase
from opennmr.processing import ft

from test_gui import (_complex, _drifting_phase_series, _load_spectrum_directly, _real_fraction,  # noqa: F401
                      qapp)


# ---- 4. fonts, xlim, findfont noise ---------------------------------------------------------

def test_font_combo_previews_only_faces_qt_knows(qapp, monkeypatch):  # noqa: F811
    """matplotlib lists faces Qt cannot load (macOS fonts not installed
    locally); asking Qt for one is what printed "Populating font family
    aliases ... Replace uses of missing font family 'Athelas'"."""
    from PySide6.QtCore import Qt
    from opennmr.notebook import fonts as nb_fonts
    requested = []
    import opennmr.gui.main_window as mw

    class SpyFont(mw.QFont):
        def __init__(self, *args, **kwargs):
            if args and isinstance(args[0], str):
                requested.append(args[0])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mw, "QFont", SpyFont)
    monkeypatch.setattr(nb_fonts, "available_fonts",
                        lambda text_only=True: ["Athelas Not Installed Anywhere", "DejaVu Sans"])
    window = MainWindow()
    combo = window.output_font_family_combo
    assert "Athelas Not Installed Anywhere" not in requested
    names = [combo.itemText(i) for i in range(combo.count())]
    assert "Athelas Not Installed Anywhere" in names            # still offered, drawn in the menu font
    index = names.index("Athelas Not Installed Anywhere")
    assert combo.itemData(index, Qt.ItemDataRole.FontRole) is None
    if "DejaVu Sans" in QFontDatabase.families():
        assert "DejaVu Sans" in requested


def test_a_one_point_spectrum_draws_without_the_identical_xlims_warning(qapp):  # noqa: F811
    ppm = np.array([4.7])
    spectrum = NMRSpectrum(
        data=np.array([1.0]), data_imag=np.array([0.0]),
        axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    window = MainWindow()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*identical low and high xlims.*")
        _load_spectrum_directly(window, spectrum)
        window.canvas.show_spectrum(spectrum)
    low, high = window.canvas.ax.get_xlim()
    assert low != high and min(low, high) < 4.7 < max(low, high)


def test_safe_xlim():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from opennmr.plotting import safe_xlim
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert safe_xlim(ax, 2.0, 2.0) == (1.5, 2.5)
        assert safe_xlim(ax, 1000.0, 1000.0) == (950.0, 1050.0)
        assert safe_xlim(ax, 1.0, 3.0) == (1.0, 3.0)
        before = ax.get_xlim()
        assert safe_xlim(ax, float("nan"), 3.0) == before        # non-finite: left alone
    plt.close(fig)


def test_the_macos_font_weight_message_is_dropped_but_other_font_messages_are_not():
    import opennmr  # noqa: F401 -- installs the filter
    logger = logging.getLogger("matplotlib.font_manager")
    noise = logger.makeRecord(logger.name, logging.WARNING, __file__, 1,
                              "findfont: Failed to find font weight normal for Apple Chancery, now using 0.",
                              (), None)
    real = logger.makeRecord(logger.name, logging.WARNING, __file__, 1,
                             "findfont: Font family 'Nope' not found.", (), None)
    assert not logger.filter(noise)
    assert logger.filter(real)     # Logger.filter returns the record on Python >= 3.12


# ---- 6. autophase each slice -------------------------------------------------------------------

def test_notebook_autophase_slices_phases_every_row_on_its_own():
    spectrum = _drifting_phase_series(n_rows=6, drift_deg=150.0)
    found = slice_phase.autophase_slices(spectrum)
    assert len(found["phase0"]) == 6 and found["phase1"] == [0.0] * 6
    for row in range(6):
        assert _real_fraction(spectrum, row) > 0.98
    # Each slice got its own: they differ along the drift.
    assert max(found["phase0"]) - min(found["phase0"]) > 90.0 or \
        max(np.abs(found["phase0"])) > 90.0
    recorded = spectrum.metadata.raw["slice_phase0"]
    assert len(recorded) == 6 and any(recorded)


def test_notebook_autophase_slices_does_not_borrow_another_slices_phase():
    spectrum = _drifting_phase_series(n_rows=4, drift_deg=0.0)
    # Row 0 well phased, row 3 off by 100 degrees: row 0 must stay put.
    spectrum.data_imag[3], spectrum.data[3] = (
        np.imag((spectrum.data[3] + 1j * spectrum.data_imag[3]) * np.exp(1j * np.deg2rad(100.0))),
        np.real((spectrum.data[3] + 1j * spectrum.data_imag[3]) * np.exp(1j * np.deg2rad(100.0))))
    row0_before = spectrum.complex_data[0].copy()
    found = slice_phase.autophase_slices(spectrum)
    assert abs(found["phase0"][0]) < 2.0
    np.testing.assert_allclose(spectrum.complex_data[0], row0_before, atol=1e-2 * np.abs(row0_before).max())
    assert abs(abs(found["phase0"][3]) - 100.0) < 3.0
    assert _real_fraction(spectrum, 3) > 0.98


def test_autophase_slices_subset_and_joint_and_errors():
    spectrum = _drifting_phase_series(n_rows=4, drift_deg=120.0)
    found = slice_phase.autophase_slices(spectrum, rows=[2])
    assert len(found["phase0"]) == 1
    assert not spectrum.metadata.raw["slice_phase0"][0] and spectrum.metadata.raw["slice_phase0"][2]
    joint = slice_phase.autophase_slices(_drifting_phase_series(n_rows=3), joint=True)
    assert len(joint["phase1"]) == 3
    one_d = NMRSpectrum(data=np.ones(8), data_imag=np.zeros(8),
                        axes=[NMRAxis(kind="frequency", ppm=np.linspace(0, 1, 8), hz=np.linspace(0, 1, 8))],
                        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    with pytest.raises(ValueError):
        slice_phase.autophase_slices(one_d)


def test_gui_each_slice_checkbox_is_separate_from_apply_to_every_slice(qapp):  # noqa: F811
    spectrum = _drifting_phase_series(drift_deg=150.0)
    window = MainWindow()
    window.show()
    assert window.autophase_each_slice_checkbox.isHidden()          # nothing selected yet
    _load_spectrum_directly(window, spectrum)
    assert not window.autophase_each_slice_checkbox.isHidden()
    assert not window.autophase_each_slice_checkbox.isChecked()

    # Apply to every slice OFF, each-slice ON: every slice is phased on its own.
    window.apply_all_slices_checkbox.setChecked(False)
    window.autophase_each_slice_checkbox.setChecked(True)
    window._on_autophase_clicked()
    result = window.session.spectra[0]
    for row in range(result.data.shape[0]):
        assert _real_fraction(result, row) > 0.98
    assert any(result.metadata.raw["slice_phase0"])


def test_gui_without_the_option_autophase_still_phases_only_the_selected_slice(qapp):  # noqa: F811
    spectrum = _drifting_phase_series(drift_deg=150.0)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.apply_all_slices_checkbox.setChecked(False)
    window.autophase_each_slice_checkbox.setChecked(False)
    last = spectrum.data.shape[0] - 1
    window.slice_spin.setValue(1)
    before_last = window.session.spectra[0].complex_data[last].copy()
    window._on_autophase_clicked()
    result = window.session.spectra[0]
    assert _real_fraction(result, 0) > 0.98
    np.testing.assert_allclose(result.complex_data[last], before_last, atol=1e-9)


def test_gui_each_slice_with_first_order_records_both(qapp):  # noqa: F811
    spectrum = _drifting_phase_series(n_rows=3, drift_deg=60.0)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.autophase_each_slice_checkbox.setChecked(True)
    window.autophase_joint_checkbox.setChecked(True)
    window._on_autophase_clicked()
    raw = window.session.spectra[0].metadata.raw
    assert len(raw["slice_phase0"]) == 3 and len(raw["slice_phase1"]) == 3


def test_the_disagreement_warning_names_the_new_option(qapp):  # noqa: F811
    spectrum = _drifting_phase_series(drift_deg=120.0)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.apply_all_slices_checkbox.setChecked(False)
    window._on_autophase_clicked()
    message = window.processing_status_label.text()
    assert "Each slice independently" in message and "Apply to every slice" in message


def test_each_slice_option_is_hidden_for_a_1d(qapp, tmp_path):  # noqa: F811
    from test_gui import _make_1d_dataset
    window = MainWindow()
    window.show()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    assert not window.autophase_each_slice_checkbox.isVisible()


# ---- 5. drift correction beyond Bruker -------------------------------------------------------

DT = 1.0e-3                    # 1 kHz spectral width
N = 2048


def _peak_hz(fid):
    freq_hz, spec = ft.fft_spectrum(fid, DT)
    return float(freq_hz[int(np.argmax(np.abs(spec)))])


def _series(n_rows=5, f0=120.0):
    t = np.arange(N) * DT
    fid = np.exp(2j * np.pi * f0 * t) * np.exp(-t / 0.3)
    return np.tile(fid, (n_rows, 1)), f0


def test_pseudo_2d_correction_moves_each_row_by_its_share_of_the_drift():
    rows, f0 = _series()
    fixed = dc.correct_pseudo_2d_fids(rows, DT, start_hz=0.0, end_hz=8.0)
    freqs = [_peak_hz(row) for row in fixed]
    resolution = 1.0 / (N * DT)
    # openNMR's frequency axis is the mirror of the raw FID's, so compare with the
    # uncorrected peak: a positive shift moves the peak to higher frequency (left in ppm)
    expected = _peak_hz(rows[0]) + np.linspace(0.0, 8.0, 5)
    np.testing.assert_allclose(freqs, expected, atol=resolution)
    assert freqs[-1] - freqs[0] == pytest.approx(8.0, abs=2 * resolution)
    # input untouched
    assert _peak_hz(rows[0]) == pytest.approx(-f0, abs=resolution)


def test_pseudo_2d_correction_sign_matches_the_bruker_convention():
    """Positive shift = peaks move to higher frequency (left on a ppm axis),
    exactly as for Bruker data: enter reference - observed."""
    rows, f0 = _series(n_rows=2)
    up = dc.correct_pseudo_2d_fids(rows, DT, 3.0, 3.0)
    down = dc.correct_pseudo_2d_fids(rows, DT, -3.0, -3.0)
    base = _peak_hz(rows[0])        # the displayed axis is the mirror of the raw FID's
    assert _peak_hz(up[0]) > base > _peak_hz(down[0])


def test_pseudo_2d_correction_equals_the_bruker_code_path_for_a_pseudo_dimension():
    rows, _ = _series(n_rows=4)
    shifts = dc.drift_profile(4, 1.0, 5.0)
    expected = dc.apply_drift_correction(rows, shifts, 1.0 / DT, [dc.IndirectDim("F1", 4, "none")],
                                         None, 0, None, 0.0, "opennmr")
    np.testing.assert_allclose(dc.correct_pseudo_2d_fids(rows, DT, 1.0, 5.0), expected)
    np.testing.assert_allclose(dc.correct_pseudo_2d_fids(rows, DT, shifts_hz=shifts), expected)


def test_pseudo_2d_shifts_and_record():
    assert dc.pseudo_2d_shifts(3, {"start_hz": 0.0, "end_hz": 2.0}).tolist() == [0.0, 1.0, 2.0]
    assert dc.pseudo_2d_shifts(4, {"start_hz": 0.0, "end_hz": 3.0, "endpoint": False}).tolist() == \
        pytest.approx([0.0, 0.75, 1.5, 2.25])
    with pytest.raises(ValueError):
        dc.pseudo_2d_shifts(3, {"shifts_hz": [1.0, 2.0]})
    record = dc.pseudo_2d_record(5, {"start_hz": 1.0, "end_hz": 2.0}, "Varian")
    assert record["start_hz"] == 1.0 and record["end_hz"] == 2.0 and record["n_points"] == 5
    assert "Varian" in record["description"] and record["citation"] == dc.CITATION


def _meta(vendor, pseudo=True, ndim=2):
    data = np.zeros((3, 8)) if ndim == 2 else np.zeros(8)
    axes = ([NMRAxis(kind="index", index=np.arange(1, 4))] if ndim == 2 else []) + [
        NMRAxis(kind="frequency", ppm=np.linspace(0, 1, 8), hz=np.linspace(0, 400, 8))]
    return NMRSpectrum(data=data, axes=axes, data_imag=np.zeros_like(data),
                       metadata=NMRMetadata(vendor=vendor, nucleus="1H", sfo1=400.0, pseudo_2d=pseudo))


# JEOL joined 2026-11-09, with its 2D reader (io.jeol.read_2d corrects a
# pseudo-2D series and refuses a genuine 2D, like the others).
@pytest.mark.parametrize("vendor", ["xpulse", "varian", "spinsolve", "jeol"])
def test_drift_support_accepts_a_pseudo_2d_from_each_supported_vendor(vendor):
    assert dc.drift_support(_meta(vendor))[0] is True


@pytest.mark.parametrize("vendor", ["xpulse", "varian", "spinsolve", "jeol"])
def test_drift_support_refuses_the_rest_with_a_reason(vendor):
    ok, why = dc.drift_support(_meta(vendor, pseudo=False))
    assert not ok and "Bruker" in why
    ok, why = dc.drift_support(_meta(vendor, ndim=1))
    assert not ok


def test_bruker_support_is_unchanged():
    spectrum = _meta("bruker")
    assert dc.drift_support(spectrum)[0] is False                 # no raw ser cached
    spectrum.metadata.raw["fid_raw"] = np.zeros((3, 8), complex)
    assert dc.drift_support(spectrum)[0] is True


def test_varian_read_2d_applies_the_correction_to_its_fids(tmp_path):
    from test_varian import _write_varian_2d
    from opennmr.io import varian
    path = _write_varian_2d(tmp_path, ni=4, n_direct=64, sw=1000.0)
    plain = varian.read_2d(path, autophase=False)
    fixed = varian.read_2d(path, autophase=False,
                           drift_correction={"start_hz": 0.0, "end_hz": 30.0})
    assert fixed.metadata.raw["drift_correction"]["end_hz"] == 30.0
    assert "drift_correction" not in plain.metadata.raw
    # Row 0 has a zero shift; the last row does not.
    np.testing.assert_allclose(fixed.complex_data[0], plain.complex_data[0], atol=1e-9)
    assert not np.allclose(fixed.complex_data[-1], plain.complex_data[-1])
    # and a 1D read through the registry just ignores the request
    one_d = varian._read_auto(_write_varian_1d_for(tmp_path), drift_correction={"start_hz": 1.0, "end_hz": 2.0},
                              autophase=False)
    assert one_d.ndim == 1


def _write_varian_1d_for(tmp_path):
    from test_varian import _write_varian_1d
    return _write_varian_1d(tmp_path, n=16, name="one.fid")


def test_xpulse_array_applies_the_correction_step_by_step(tmp_path):
    from test_xpulse import _make_array_experiment
    from opennmr.io import xpulse
    path = _make_array_experiment(tmp_path, n_steps=4)
    plain = xpulse.read_array(path, autophase=False, baseline_correct=False)
    fixed = xpulse.read_array(path, autophase=False, baseline_correct=False,
                              drift_correction={"start_hz": 0.0, "end_hz": 6.0})
    assert fixed.metadata.raw["drift_correction"]["end_hz"] == 6.0
    assert fixed.metadata.raw["fid_raw"].shape == plain.metadata.raw["fid_raw"].shape
    np.testing.assert_allclose(fixed.metadata.raw["fid_raw"][0], plain.metadata.raw["fid_raw"][0])
    assert not np.allclose(fixed.metadata.raw["fid_raw"][-1], plain.metadata.raw["fid_raw"][-1])
    # the stored raw FID is what the correction produced, so a rebuild keeps it
    dt = plain.metadata.raw["fid_dt"]
    expected = dc.correct_pseudo_2d_fids(plain.metadata.raw["fid_raw"], dt, 0.0, 6.0)
    np.testing.assert_allclose(fixed.metadata.raw["fid_raw"], expected, atol=1e-9)


def test_xpulse_single_file_ignores_the_request_and_ntuples_2d_refuses(tmp_path, monkeypatch):
    from test_xpulse import _make_array_experiment
    from opennmr.io import xpulse
    path = _make_array_experiment(tmp_path, n_steps=2)
    jdx = xpulse._find_array_step_jdx(xpulse._find_array_dirs(path)[0][1])
    one = xpulse._read_auto(jdx, drift_correction={"start_hz": 1.0, "end_hz": 2.0},
                            autophase=False, baseline_correct=False)
    assert one.ndim == 1
    monkeypatch.setattr(xpulse, "_is_ntuples_2d_file", lambda p: True)
    with pytest.raises(ValueError, match="Bruker"):
        xpulse._read_auto(jdx, drift_correction={"start_hz": 1.0, "end_hz": 2.0})


def test_gui_applies_and_removes_a_drift_correction_on_an_xpulse_array(qapp, tmp_path):  # noqa: F811
    from test_xpulse import _make_array_experiment
    window = MainWindow()
    window._load(_make_array_experiment(tmp_path, n_steps=4), vendor="xpulse")
    index = window._selected_index()
    before = window.session.spectra[index].complex_data.copy()
    assert window._apply_drift_correction_to_selected({"start_hz": 0.0, "end_hz": 5.0}) is None
    spectrum = window.session.spectra[index]
    assert spectrum.metadata.raw["drift_correction"]["end_hz"] == 5.0
    assert not np.allclose(spectrum.complex_data, before)
    assert window.drift_panel.apply_button.isEnabled()
    assert window._apply_drift_correction_to_selected(None) is None
    assert "drift_correction" not in window.session.spectra[index].metadata.raw


def test_gui_refuses_a_genuine_2d_from_another_vendor_with_a_reason(qapp):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _meta("xpulse", pseudo=False))
    message = window._apply_drift_correction_to_selected({"start_hz": 0.0, "end_hz": 1.0})
    assert message and "Bruker" in message
    assert not window.drift_panel.apply_button.isEnabled()
    assert "pseudo-2D" in window.drift_panel.target_label.text() or "Bruker" in window.drift_panel.target_label.text()
