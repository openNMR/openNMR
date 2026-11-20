"""Notes 8.7: Amplitude as a peak-table input (start, Fixed, bounds)."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import fitting
from opennmr.processing.fitting import gaussian


def _two_overlapping(scale=1.0, a1=1.0, a2=0.3, x1=5.0, x2=5.08, fwhm=0.06):
    ppm = np.linspace(4.0, 6.0, 1500)
    y = scale * (gaussian(ppm, x1, fwhm, a1) + gaussian(ppm, x2, fwhm, a2))
    md = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)], metadata=md)


# ---- notebook level -----------------------------------------------------------------

def test_a_lower_amplitude_bound_above_zero_survives_positive_intensity():
    peaks = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.1, "bounds": {"amplitude": (2.0, 9.0)}}]
    lower, upper = fitting.build_bounds(peaks, positive_intensity=True)
    assert lower[2] == 2.0 and upper[2] == 9.0
    lower, _ = fitting.build_bounds([{"type": "gaussian", "shift": 5.0, "fwhm": 0.1,
                                      "bounds": {"amplitude": (-3.0, 9.0)}}], positive_intensity=True)
    assert lower[2] == 0.0                        # a negative one is still floored at zero


def test_user_amplitudes_are_brought_into_the_normalised_fit_units():
    peaks = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.1, "A_init": 50.0,
              "bounds": {"amplitude": (10.0, np.inf)}}]
    scaled = fitting.peak_defs_in_fit_units(peaks, 10.0)
    assert scaled[0]["A_init"] == 5.0 and scaled[0]["bounds"]["amplitude"] == (1.0, np.inf)
    assert peaks[0]["A_init"] == 50.0             # the caller's own definitions are untouched
    assert fitting.peak_defs_in_fit_units(peaks, 1.0) is peaks


@pytest.mark.parametrize("normalised", [True, False])
def test_a_fixed_weak_peak_stays_at_the_given_amplitude(normalised):
    spec = _two_overlapping(scale=1000.0)
    peaks = [
        {"type": "gaussian", "shift": 5.0, "fwhm": 0.06},
        {"type": "gaussian", "shift": 5.08, "fwhm": 0.06, "A_init": 250.0, "fixed": {"A": True}},
    ]
    popt, _pcov, _free, scale, _x_ref = fitting.fit_spectrum(spec, peaks, use_normalised=normalised)
    assert popt[5] * scale == pytest.approx(250.0, rel=1e-9)   # held, in the spectrum's own units
    # the strong peak absorbs the rest: 1000 true, the held peak is 50 below its true 300
    assert popt[2] * scale > 900.0


@pytest.mark.parametrize("normalised", [True, False])
def test_a_start_value_does_not_change_where_the_fit_converges(normalised):
    spec = _two_overlapping(scale=1000.0)
    base = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.06},
            {"type": "gaussian", "shift": 5.08, "fwhm": 0.06}]
    seeded = [dict(base[0], A_init=900.0), dict(base[1], A_init=200.0)]
    a = fitting.fit_spectrum(spec, base, use_normalised=normalised)
    b = fitting.fit_spectrum(spec, seeded, use_normalised=normalised)
    assert a[0][2] * a[3] == pytest.approx(1000.0, rel=1e-3)
    assert b[0][2] * b[3] == pytest.approx(a[0][2] * a[3], rel=1e-3)
    assert b[0][5] * b[3] == pytest.approx(300.0, rel=1e-2)


def test_an_amplitude_bound_means_the_same_thing_normalised_or_not():
    # A bound is in the spectrum's units. Normalised, a 500 cap on a 2000-high peak
    # used to be compared with an amplitude of ~1 and never bind.
    spec = _two_overlapping(scale=2000.0, a2=0.0)
    peaks = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.06, "bounds": {"amplitude": (-np.inf, 500.0)}}]
    for normalised in (True, False):
        popt, _p, _f, scale, _x = fitting.fit_spectrum(spec, peaks, use_normalised=normalised)
        assert popt[2] * scale == pytest.approx(500.0, rel=1e-3)


# ---- GUI ----------------------------------------------------------------------------

from opennmr.gui.main_window import MainWindow  # noqa: E402
from test_gui import qapp  # noqa: E402,F401


def test_peak_table_has_an_amplitude_column_that_is_blank_by_default(qapp):
    window = MainWindow()
    assert window._PEAK_TABLE_COLUMNS[window._PEAK_AMPLITUDE_COL] == "Amplitude"
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    assert window.peak_table.columnCount() == len(window._PEAK_TABLE_COLUMNS)
    assert window.peak_table.item(0, window._PEAK_AMPLITUDE_COL).text() == ""
    assert "A_init" not in window._read_peak_defs()[0]


def test_a_typed_amplitude_becomes_the_peaks_starting_value(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0, amplitude=2.5)
    window._add_peak_row(shift=6.0, fwhm=0.05, eta=0.0)
    defs = window._read_peak_defs()
    assert defs[0]["A_init"] == 2.5
    assert "A_init" not in defs[1]


def test_amplitude_with_fixed_is_held(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0, amplitude=2.5, per_param_fixed={"A": True})
    peak = window._read_peak_defs()[0]
    assert peak["A_init"] == 2.5 and peak["fixed"]["A"] is True
    window.peak_table.item(0, 3).setCheckState(window.peak_table.item(0, 3).checkState().__class__.Checked)
    assert window._read_peak_defs()[0]["fixed"]["A"] is True


def test_a_bad_amplitude_is_reported_by_row(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window.peak_table.item(0, window._PEAK_AMPLITUDE_COL).setText("lots")
    with pytest.raises(ValueError, match="row 1.*Amplitude"):
        window._read_peak_defs()


def test_an_amplitude_outside_its_own_bounds_is_refused_unless_fixed(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0, amplitude=9.0, bounds={"amplitude": (0.0, 5.0)})
    with pytest.raises(ValueError, match="outside its bounds"):
        window._read_peak_defs()
    window.peak_table.item(0, 0).setData(window._PEAK_FIXED_ITEM_ROLE, {"A": True})
    assert window._read_peak_defs()[0]["A_init"] == 9.0      # fixed: the bounds do not apply


def test_amplitude_survives_the_saved_analysis_state(qapp):
    window = MainWindow()
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0, amplitude=2.5)
    state = window._capture_analysis_state()
    assert state["peaks"][0][window._PEAK_AMPLITUDE_COL] == "2.5"
    window.peak_table.setRowCount(0)

    class _Spec:                       # only what _restore_analysis_state reads
        metadata = type("M", (), {"raw": {"analysis_state": state}})()
        ndim = 1
    window._restore_analysis_state(_Spec())
    assert window.peak_table.item(0, window._PEAK_AMPLITUDE_COL).text() == "2.5"
    assert window._read_peak_defs()[0]["A_init"] == 2.5
