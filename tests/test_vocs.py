import numpy as np

from nmrsuite.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from nmrsuite.processing import vocs


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
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1, raw={"acqus": {"P1": p1}})
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
    spec = _make_1d()
    try:
        vocs.apply_vocs(spec, pulse="P9")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_vocs_sum_combines_multiple_spectra():
    specs = [_make_1d(center_ppm=c) for c in (4.0, 5.0, 6.0)]
    result = vocs.vocs_sum(specs, npoints=512)
    assert len(result["ppm"]) == 512
    assert result["interpolated_slices"].shape == (3, 512)
    assert len(result["vocs_sum"]) == 512
