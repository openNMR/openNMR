import os

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import vocs

_VOCS_DATA = os.path.join(
    "test_data", "VOCS_example_Data", "15_02_26_Li2MnO3_O17_1_3mm",
)


def test_pulse_to_w1_matches_docstring_example():
    assert vocs.pulse_to_w1(5) == 50000.0


def test_vocs_power_windowed_zero_outside_bandwidth():
    freq_hz = np.linspace(-100000, 100000, 2001)
    power = vocs.vocs_power(freq_hz, w1=1000.0, window_filter=True)
    x = (freq_hz - freq_hz[len(freq_hz) // 2]) / 1000.0
    assert np.all(power[np.abs(x) >= 1.73] == 0)


def test_vocs_power_peaks_near_centre():
    freq_hz = np.linspace(-5000, 5000, 501)
    power = vocs.vocs_power(freq_hz, w1=1000.0, window_filter=False)
    assert np.argmax(power) == np.argmin(np.abs(freq_hz))


def _make_1d(n=64, center_ppm=5.0, sfo1=400.0, p1=5.0):
    ppm = np.linspace(center_ppm - 2, center_ppm + 2, n)
    hz = ppm * sfo1
    data = np.exp(-((ppm - center_ppm) ** 2) / 0.05)
    # `P` as a real indexed ARRAY, the way every Bruker acqus actually
    # records it -- there is no "P1" key in a real file. This fixture
    # used to write {"P1": p1}, which is why the whole VOCS path could
    # be green in tests while failing on every real dataset with "P1
    # not found in this spectrum's acquisition parameters" (real user
    # report). A fixture that doesn't match the file format tests
    # nothing but itself.
    acqus = {"P": [0.0, p1, 2 * p1] + [0.0] * 61}
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1, raw={"acqus": acqus})
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=hz)
    return NMRSpectrum(data=data, axes=[axis], metadata=metadata)


def test_apply_vocs_scales_data_and_records_scaling():
    spec = _make_1d()
    corrected = vocs.apply_vocs(spec, pulse="P1")
    assert "vocs_scaling" in corrected.metadata.raw
    assert len(corrected.data) == len(spec.data)
    # Some scaling was actually applied (not a no-op).
    assert not np.allclose(corrected.data, spec.data)


def test_apply_vocs_missing_pulse_raises():
    # P9 exists in a real acqus but is 0 (unused) -- it must raise a
    # ValueError naming the problem, not reach pulse_to_w1 and come
    # back as a bare ZeroDivisionError.
    spec = _make_1d()
    try:
        vocs.apply_vocs(spec, pulse="P9")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_pulse_length_resolves_an_indexed_p_array():
    # The actual bug: a real acqus records P as an array, so a direct
    # `"P1" in acqus` lookup is always False and VOCS raised on every
    # real dataset. Resolution goes through acquisition_constants,
    # which expands P/D/CNST/L/GPZ into p1/d1/cnst2/l4/gpz1 names.
    spec = _make_1d(p1=4.25)
    assert "P1" not in spec.metadata.raw["acqus"]   # the real file's shape
    assert vocs._pulse_length_us(spec, "P1") == 4.25
    assert vocs._pulse_length_us(spec, "p1") == 4.25  # case-insensitive
    assert vocs._pulse_length_us(spec, "P2") == 8.50


def test_pulse_length_error_names_the_pulses_that_are_set():
    # A Bruker acqus always carries all 64 P slots and most are 0, so
    # the message must name the two or three actually in use rather
    # than listing every key.
    spec = _make_1d(p1=4.25)
    try:
        vocs._pulse_length_us(spec, "P9")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert "P1 (4.25 us)" in message
    assert "P2 (8.5 us)" in message
    assert "P3" not in message  # zero/unset, so not offered
    assert "recorded as 0" in message  # says *why* P9 is unusable


def test_pulse_length_on_a_spectrum_with_no_acqus_says_so():
    spec = _make_1d()
    spec.metadata.raw.pop("acqus")
    try:
        vocs._pulse_length_us(spec, "P1")
    except ValueError as exc:
        assert "no recorded acquisition parameters" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_vocs_sum_combines_multiple_spectra():
    specs = [_make_1d(center_ppm=c) for c in (4.0, 5.0, 6.0)]
    result = vocs.vocs_sum(specs, npoints=512)
    assert len(result["ppm"]) == 512
    assert result["interpolated_slices"].shape == (3, 512)
    assert len(result["vocs_sum"]) == 512


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_VOCS_DATA, "12")),
    reason="needs the real VOCS_example_Data dataset",
)
def test_vocs_on_the_real_offset_series():
    # The dataset the bug was reported on: 17O on Li2MnO3, experiments
    # 12-15, four offset acquisitions covering overlapping excitation
    # windows (O1 = +671, +400, -136, -407 kHz) -- exactly what VOCS
    # exists to recombine. Every one of these raised "P1 not found in
    # this spectrum's acquisition parameters" before the fix.
    from opennmr.io import bruker

    specs = [bruker.read_1d(os.path.join(_VOCS_DATA, e)) for e in ("12", "13", "14", "15")]
    assert {s.metadata.nucleus for s in specs} == {"17O"}
    # Genuinely different offsets -- otherwise this is not a VOCS set.
    assert len({round(s.metadata.o1) for s in specs}) == 4
    # The real file's shape, which the old direct lookup could not read.
    assert "P1" not in specs[0].metadata.raw["acqus"]

    for spec in specs:
        corrected = vocs.apply_vocs(spec, pulse="P1")
        assert corrected.data.shape == spec.data.shape
        assert np.isfinite(corrected.data).all()

    result = vocs.vocs_sum(specs, pulse="P1")
    assert np.isfinite(result["vocs_sum"]).all()
    assert result["interpolated_slices"].shape[0] == 4
    # The combined axis has to span every contributing offset, not just
    # one window -- that recombination is the whole point of VOCS.
    for spec in specs:
        assert result["ppm"].min() <= spec.axes[0].ppm.max()
        assert result["ppm"].max() >= spec.axes[0].ppm.min()
