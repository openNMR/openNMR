"""Field drift seen along the slices and removed, for any vendor."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import field_drift as fd

PPM = np.linspace(10, 0, 4096)
SFO = 400.0
N_ROWS = 24


def _series(drift_hz, real_only=False, vendor="xpulse"):
    rows = []
    for d in drift_hz:
        x = (PPM - (5.0 + d / SFO)) / 0.01
        rows.append((1 + 1j * x) / (1 + x ** 2))
    data = np.array(rows)
    return NMRSpectrum(
        data=data.real, data_imag=None if real_only else data.imag,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(drift_hz) + 1)),
              NMRAxis(kind="frequency", ppm=PPM, hz=PPM * SFO)],
        metadata=NMRMetadata(vendor=vendor, nucleus="1H", sfo1=SFO, pseudo_2d=True))


def _spread_hz(spectrum):
    hz = fd.track_peak(spectrum, (4.9, 5.1))["hz"]
    return float(np.max(hz) - np.min(hz))


def test_track_peak_follows_the_drift():
    drift = np.linspace(0, 3.0, N_ROWS)
    track = fd.track_peak(_series(drift), (4.9, 5.1))
    np.testing.assert_allclose(track["hz"] - track["hz"][0], drift, atol=0.1)
    assert list(track["row"][:3]) == [0, 1, 2]


def test_centroid_method_agrees():
    drift = np.linspace(0, 3.0, N_ROWS)
    track = fd.track_peak(_series(drift), (4.9, 5.1), method="centroid")
    np.testing.assert_allclose(track["hz"] - track["hz"][0], drift, atol=0.3)


@pytest.mark.parametrize("real_only", [False, True])
def test_linear_drift_is_removed(real_only):
    drift = np.linspace(0, 3.0, N_ROWS)
    sp = _series(drift, real_only=real_only)
    out = fd.correct_field_drift(sp, (4.9, 5.1), model="linear")
    assert out["fit"]["params"]["slope_hz_per_slice"] == pytest.approx(3.0 / (N_ROWS - 1), rel=0.05)
    assert _spread_hz(sp) < 0.1
    assert sp.metadata.raw["field_drift"]["model"] == "linear"


def test_curved_drift_needs_more_than_a_line():
    x = np.arange(N_ROWS)
    drift = 0.004 * x ** 2
    line = _series(drift)
    fd.correct_field_drift(line, (4.9, 5.1), model="linear")
    poly = _series(drift)
    fd.correct_field_drift(poly, (4.9, 5.1), model="polynomial", degree=2)
    assert _spread_hz(poly) < 0.1 < _spread_hz(line)


def test_sinusoidal_drift():
    x = np.arange(N_ROWS)
    drift = 0.05 * x + 1.5 * np.sin(2 * np.pi * x / 12.0)
    sp = _series(drift)
    out = fd.correct_field_drift(sp, (4.9, 5.1), model="sinusoidal")
    assert out["fit"]["params"]["period_slices"] == pytest.approx(12.0, rel=0.1)
    assert _spread_hz(sp) < 0.3


def test_automatic_spline_follows_a_smooth_curve():
    x = np.arange(N_ROWS)
    drift = 2.0 * np.tanh((x - 10) / 5.0)
    sp = _series(drift)
    fd.correct_field_drift(sp, (4.9, 5.1), model="spline")
    assert _spread_hz(sp) < 0.4


def test_spline_through_chosen_slices():
    x = np.arange(N_ROWS)
    drift = 2.0 * np.tanh((x - 10) / 5.0)
    sp = _series(drift)
    out = fd.correct_field_drift(sp, (4.9, 5.1), model="spline_points",
                                 node_x=[0, 6, 10, 14, 23])
    assert out["fit"]["params"]["nodes"] == [0.0, 6.0, 10.0, 14.0, 23.0]
    assert _spread_hz(sp) < 0.5


def test_remove_restores_the_original_and_apply_twice_does_not_stack():
    drift = np.linspace(0, 3.0, N_ROWS)
    sp = _series(drift)
    before = sp.data.copy()
    fd.correct_field_drift(sp, (4.9, 5.1), model="linear")
    fd.correct_field_drift(sp, (4.9, 5.1), model="linear")   # second call starts from the original
    assert _spread_hz(sp) < 0.1
    assert fd.remove_drift(sp) is True
    np.testing.assert_array_equal(sp.data, before)
    assert "field_drift" not in sp.metadata.raw
    assert fd.remove_drift(sp) is False


def test_reference_choice():
    drift = np.linspace(0, 3.0, N_ROWS)
    sp = _series(drift)
    out = fd.correct_field_drift(sp, (4.9, 5.1), model="linear", reference="last")
    hz = fd.track_peak(sp, (4.9, 5.1))["hz"]
    assert hz[0] == pytest.approx(hz[-1], abs=0.1)
    assert out["fit"]["shifts_hz"][-1] == pytest.approx(0.0, abs=0.05)


def test_record_survives_a_reload_via_reapply():
    drift = np.linspace(0, 3.0, N_ROWS)
    sp = _series(drift)
    fd.correct_field_drift(sp, (4.9, 5.1), model="linear")
    record = dict(sp.metadata.raw["field_drift"])
    fresh = _series(drift)
    fresh.metadata.raw["field_drift"] = record
    assert fd.reapply_record(fresh) is True
    assert _spread_hz(fresh) < 0.1
    assert fd.reapply_record(fresh) is False


def test_wrong_inputs():
    sp = _series(np.zeros(N_ROWS))
    with pytest.raises(ValueError):
        fd.track_peak(sp, (5.0, 5.0001))
    with pytest.raises(ValueError):
        fd.fit_drift([0, 1, 2], [0, 1, 2], model="nope")
    with pytest.raises(ValueError):
        fd.apply_drift(sp, [0.0, 1.0])


def test_plot_draws_points_and_curve():
    import matplotlib
    matplotlib.use("Agg")
    drift = np.linspace(0, 3.0, N_ROWS)
    sp = _series(drift)
    out = fd.correct_field_drift(sp, (4.9, 5.1), apply=False)
    ax = fd.plot_drift(out["track"], out["fit"])
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Slice"


# ---- the Field Drift Correction panel ---------------------------------------------------------

from test_gui import qapp, _load_spectrum_directly  # noqa: E402,F401


def test_panel_tracks_fits_and_corrects_a_series(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    sp = _series(np.linspace(0, 3.0, N_ROWS), vendor="xpulse")
    _load_spectrum_directly(window, sp)
    panel = window.drift_panel
    panel.refresh()
    assert panel.track_button.isEnabled()
    panel.track_hi_spin.setValue(5.1)
    panel.track_lo_spin.setValue(4.9)
    panel.track_button.click()
    assert panel.track is not None and panel.fit is not None
    assert "Linear" in panel.track_label.text()
    assert panel.drift_apply_button.isEnabled()
    panel.drift_model_combo.setCurrentIndex(panel.drift_model_combo.findData("polynomial"))
    assert not panel.drift_degree_spin.isHidden()
    panel.drift_model_combo.setCurrentIndex(panel.drift_model_combo.findData("linear"))
    panel.drift_apply_button.click()
    spectrum = window.session.spectra[window._selected_index()]
    assert _spread_hz(spectrum) < 0.15
    assert spectrum.metadata.raw["field_drift"]["model"] == "linear"
    # Look at it again: the original drift is still what is plotted, the corrected one is drawn too.
    panel.track_button.click()
    assert panel.fit["rms"] < 0.2
    panel.drift_remove_button.click()
    assert "field_drift" not in spectrum.metadata.raw
    assert _spread_hz(spectrum) > 2.5
    window.close()


def test_panel_spline_points_can_be_picked_on_the_plot(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _series(2.0 * np.tanh((np.arange(N_ROWS) - 10) / 5.0)))
    panel = window.drift_panel
    panel.refresh()
    panel.track_hi_spin.setValue(5.1)
    panel.track_lo_spin.setValue(4.9)
    panel.track_button.click()
    panel.drift_model_combo.setCurrentIndex(panel.drift_model_combo.findData("spline_points"))
    assert not panel.drift_nodes_edit.isHidden()
    panel.drift_pick_button.setChecked(True)
    for x in (1, 8, 14, 24):
        panel.on_canvas_pick(float(x))
    assert panel.drift_nodes_edit.text() == "1, 8, 14, 24"
    assert panel.fit["params"]["nodes"] == [0.0, 7.0, 13.0, 23.0]
    window.close()


def test_drift_survives_a_reprocess_of_the_rows(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    sp = _series(np.linspace(0, 3.0, N_ROWS))
    _load_spectrum_directly(window, sp)
    fd.correct_field_drift(sp, (4.9, 5.1), model="linear")
    record = sp.metadata.raw["field_drift"]
    assert fd.record_shifts(sp) is not None and len(record["shifts_hz"]) == N_ROWS
    # A pipeline that rebuilds the uncorrected rows puts the shifts back on.
    original = sp.metadata.raw["_field_drift_original"]
    rebuilt = original[0] + 1j * original[1]
    shifted = fd.shift_rebuilt_rows(sp, rebuilt, PPM)
    sp.data, sp.data_imag = np.real(shifted), np.imag(shifted)
    assert _spread_hz(sp) < 0.15
    window.close()
